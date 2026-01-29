from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRClassDef,
    PyIRFile,
    PyIRIf,
    PyIRNode,
    PyIRVarUse,
    PyIRWith,
)
from ..analysis.symlinks import PyIRSymLink, SymbolId, SymLinkContext
from ..macro_utils import MacroUtils

_DIRECTIVE_KIND: Literal["with_condition"] = "with_condition"


@dataclass(frozen=True)
class _WithCondClassInfo:
    sym_id: int
    name: str
    file: Path | None
    line: int | None
    offset: int | None


class RewriteWithConditionPass:
    """
    Обработка @CompilerDirective.with_condition().

    Поддерживает:
        with Class.flag:
            body

        with FLAG:
            body

    где FLAG - экспортированный символ модуля,
    в котором объявлен класс с @with_condition.
    """

    _CTX_SEEN = "_withcond_seen_paths"
    _CTX_DONE = "_withcond_done"
    _CTX_CLASSES = "_withcond_classes"  # class_sym_id -> _WithCondClassInfo
    _CTX_FILES = "_withcond_files"  # list[PyIRFile]
    _CTX_EXPORT_OWNER = "_withcond_export_owner"  # export_sym_id -> class_sym_id

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        MacroUtils.ensure_pass_ctx(
            ctx,
            **{
                RewriteWithConditionPass._CTX_SEEN: set(),
                RewriteWithConditionPass._CTX_DONE: False,
                RewriteWithConditionPass._CTX_CLASSES: {},
                RewriteWithConditionPass._CTX_FILES: [],
                RewriteWithConditionPass._CTX_EXPORT_OWNER: {},
            },
        )

        if ir.path is not None:
            getattr(ctx, RewriteWithConditionPass._CTX_SEEN).add(ir.path.resolve())

        files: list[PyIRFile] = getattr(ctx, RewriteWithConditionPass._CTX_FILES)
        if ir not in files:
            files.append(ir)

        RewriteWithConditionPass._collect_marked_classes(ir, ctx)

        if MacroUtils.ready_to_finalize(
            ctx,
            seen_key=RewriteWithConditionPass._CTX_SEEN,
            done_key=RewriteWithConditionPass._CTX_DONE,
        ):
            classes: dict[int, _WithCondClassInfo] = getattr(
                ctx, RewriteWithConditionPass._CTX_CLASSES
            )

            RewriteWithConditionPass._build_export_owner_index(ctx, classes)

            for f in files:
                f.body[:] = RewriteWithConditionPass._rewrite_stmt_list(
                    f.body, ctx=ctx, classes=classes, file=f.path
                )

    @staticmethod
    def _class_symbol_id(сcls: PyIRClassDef, ctx: SymLinkContext) -> int | None:
        decl_scope = ctx.node_scope[ctx.uid(сcls)]
        link = ctx.resolve(scope=decl_scope, name=сcls.name)
        return None if link is None else link.target.value

    @staticmethod
    def _collect_marked_classes(ir: PyIRFile, ctx: SymLinkContext) -> None:
        classes: dict[int, _WithCondClassInfo] = getattr(
            ctx, RewriteWithConditionPass._CTX_CLASSES
        )

        def make_info(node: PyIRClassDef, sym_id: int) -> _WithCondClassInfo:
            sym = ctx.symbols.get(SymbolId(sym_id))
            return _WithCondClassInfo(
                sym_id=sym_id,
                name=node.name,
                file=sym.decl_file if sym else ir.path,
                line=sym.decl_line if sym else node.line,
                offset=node.offset,
            )

        MacroUtils.collect_marked_defs(
            ir,
            ctx,
            node_type=PyIRClassDef,
            kind=_DIRECTIVE_KIND,
            out=classes,
            make_info=make_info,  # type: ignore
            get_symbol_id=RewriteWithConditionPass._class_symbol_id,  # type: ignore
        )

    @staticmethod
    def _build_export_owner_index(
        ctx: SymLinkContext,
        classes: dict[int, _WithCondClassInfo],
    ) -> None:
        ctx.ensure_module_index()

        export_owner: dict[int, int] = getattr(
            ctx, RewriteWithConditionPass._CTX_EXPORT_OWNER
        )
        export_owner.clear()

        for cls_id, info in classes.items():
            if info.file is None:
                continue

            module = ctx.module_name_by_path.get(info.file.resolve())
            if module is None:
                continue

            exports = ctx.module_exports.get(module) or {}
            for _, sym in exports.items():
                export_owner[sym.value] = cls_id

    @staticmethod
    def _attr_root(expr: PyIRAttribute) -> PyIRNode:
        cur: PyIRNode = expr
        while isinstance(cur, PyIRAttribute):
            cur = cur.value

        return cur

    @staticmethod
    def _symbol_id_of_root(node: PyIRNode, ctx: SymLinkContext) -> int | None:
        if isinstance(node, PyIRSymLink):
            return node.symbol_id

        if isinstance(node, PyIRVarUse):
            uid = ctx.uid(node)
            link = ctx.var_links.get(uid)
            return None if link is None else link.target.value

        return None

    @staticmethod
    def _is_with_condition_expr(
        expr: PyIRNode,
        *,
        ctx: SymLinkContext,
        classes: dict[int, _WithCondClassInfo],
    ) -> bool:
        export_owner: dict[int, int] = getattr(
            ctx, RewriteWithConditionPass._CTX_EXPORT_OWNER
        )

        if isinstance(expr, PyIRAttribute):
            root = RewriteWithConditionPass._attr_root(expr)
            rid = RewriteWithConditionPass._symbol_id_of_root(root, ctx)
            return rid is not None and rid in classes

        if isinstance(expr, PyIRSymLink):
            owner = export_owner.get(expr.symbol_id)
            return owner is not None and owner in classes

        if isinstance(expr, PyIRVarUse):
            rid = RewriteWithConditionPass._symbol_id_of_root(expr, ctx)
            if rid is None:
                return False

            owner = export_owner.get(rid)
            return owner is not None and owner in classes

        return False

    @staticmethod
    def _rewrite_stmt_list(
        stmts: list[PyIRNode],
        *,
        ctx: SymLinkContext,
        classes: dict[int, _WithCondClassInfo],
        file: Path | None,
    ) -> list[PyIRNode]:
        out: list[PyIRNode] = []

        for st in stmts:
            for child in MacroUtils.statement_lists(st):
                child[:] = RewriteWithConditionPass._rewrite_stmt_list(
                    child, ctx=ctx, classes=classes, file=file
                )

            if isinstance(st, PyIRWith):
                repl = RewriteWithConditionPass._rewrite_with(
                    st, ctx=ctx, classes=classes
                )
                if repl is not None:
                    out.extend(repl)
                    continue

            out.append(st)

        return out

    @staticmethod
    def _rewrite_with(
        w: PyIRWith,
        *,
        ctx: SymLinkContext,
        classes: dict[int, _WithCondClassInfo],
    ) -> list[PyIRNode] | None:
        conds: list[PyIRNode] = []

        for item in w.items:
            expr = item.context_expr
            if not RewriteWithConditionPass._is_with_condition_expr(
                expr, ctx=ctx, classes=classes
            ):
                return None
            conds.append(expr)

        if not conds:
            return None

        inner = list(w.body)
        scope = ctx.node_scope[ctx.uid(w)]

        for cond in reversed(conds):
            n_if = PyIRIf(
                line=w.line,
                offset=w.offset,
                test=cond,
                body=inner,  # type: ignore
                orelse=[],
            )
            ctx.node_scope[ctx.uid(n_if)] = scope
            inner = [n_if]

        return inner  # type: ignore

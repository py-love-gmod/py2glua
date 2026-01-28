from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from ....._cli.logging_setup import exit_with_code
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
)
from ..analysis.symlinks import PyIRSymLink, SymbolId, SymLinkContext


@dataclass
class _WithCondClassInfo:
    sym_id: int
    name: str
    file: Path | None
    line: int | None
    offset: int | None


class RewriteWithConditionPass:
    """
    Обработка @CompilerDirective.with_condition().

    Поддерживаемые конструкции:

        with Class.flag:
            body

    Также поддерживаются схлопнутые экспорты
    (например, после gmod_special_enum + collapse атрибутов):

        with FLAG:
            body

    если FLAG — это экспортируемый символ из модуля,
    в котором объявлен помеченный класс.

    Преобразование в вложенные if-блоки:

        with A.x, A.y:
            body
            
    ->
    
        if A.x:
            if A.y:
                body

    Правила:
      - Конструкция "with ... as name" запрещена.
      - Best-effort: переписываем только те with-блоки,
        для которых можем ДОКАЗАТЬ, что это with_condition.
    """

    _CTX_SEEN = "_withcond_seen_paths"
    _CTX_DONE = "_withcond_done"
    _CTX_CLASSES = "_withcond_classes"  # class_sym_id -> _WithCondClassInfo
    _CTX_FILES = "_withcond_files"  # list[PyIRFile]
    _CTX_EXPORT_OWNER = (
        "_withcond_export_owner"  # export_symbol_id(int) -> class_sym_id(int)
    )

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        RewriteWithConditionPass._ensure_ctx(ctx)

        if ir.path is not None:
            getattr(ctx, RewriteWithConditionPass._CTX_SEEN).add(ir.path.resolve())

        files: list[PyIRFile] = getattr(ctx, RewriteWithConditionPass._CTX_FILES)
        if ir not in files:
            files.append(ir)

        RewriteWithConditionPass._collect_marked_classes(ir, ctx)
        RewriteWithConditionPass._maybe_finalize(ctx)

    @staticmethod
    def _ensure_ctx(ctx: SymLinkContext) -> None:
        if not hasattr(ctx, RewriteWithConditionPass._CTX_SEEN):
            setattr(ctx, RewriteWithConditionPass._CTX_SEEN, set())

        if not hasattr(ctx, RewriteWithConditionPass._CTX_DONE):
            setattr(ctx, RewriteWithConditionPass._CTX_DONE, False)

        if not hasattr(ctx, RewriteWithConditionPass._CTX_CLASSES):
            setattr(ctx, RewriteWithConditionPass._CTX_CLASSES, {})

        if not hasattr(ctx, RewriteWithConditionPass._CTX_FILES):
            setattr(ctx, RewriteWithConditionPass._CTX_FILES, [])

        if not hasattr(ctx, RewriteWithConditionPass._CTX_EXPORT_OWNER):
            setattr(ctx, RewriteWithConditionPass._CTX_EXPORT_OWNER, {})

    @staticmethod
    def _maybe_finalize(ctx: SymLinkContext) -> None:
        if getattr(ctx, RewriteWithConditionPass._CTX_DONE):
            return

        all_paths = {p.resolve() for p in getattr(ctx, "_all_paths", set()) if p}
        seen = getattr(ctx, RewriteWithConditionPass._CTX_SEEN)
        if all_paths and seen != all_paths:
            return

        setattr(ctx, RewriteWithConditionPass._CTX_DONE, True)

        classes: dict[int, _WithCondClassInfo] = getattr(
            ctx, RewriteWithConditionPass._CTX_CLASSES
        )
        files: list[PyIRFile] = getattr(ctx, RewriteWithConditionPass._CTX_FILES)

        RewriteWithConditionPass._build_export_owner_index(ctx, classes)

        for ir in files:
            ir.body[:] = RewriteWithConditionPass._rewrite_stmt_list(
                ir.body, ctx=ctx, classes=classes, file=ir.path
            )

    @staticmethod
    def _decorator_kind(expr: PyIRNode) -> Literal["with_condition"] | None:
        if isinstance(expr, PyIRAttribute) and expr.attr == "with_condition":
            return "with_condition"

        if isinstance(expr, PyIRCall):
            f = expr.func
            if isinstance(f, PyIRAttribute) and f.attr == "with_condition":
                return "with_condition"
            if isinstance(f, PyIRSymLink) and f.name == "with_condition":
                return "with_condition"

        if isinstance(expr, PyIRSymLink) and expr.name == "with_condition":
            return "with_condition"

        return None

    @staticmethod
    def _class_symbol_id(classs: PyIRClassDef, ctx: SymLinkContext) -> int | None:
        decl_scope = ctx.node_scope[ctx.uid(classs)]
        link = ctx.resolve(scope=decl_scope, name=classs.name)
        return None if link is None else link.target.value

    @staticmethod
    def _collect_marked_classes(ir: PyIRFile, ctx: SymLinkContext) -> None:
        classes: dict[int, _WithCondClassInfo] = getattr(
            ctx, RewriteWithConditionPass._CTX_CLASSES
        )

        for node in ir.walk():
            if not isinstance(node, PyIRClassDef):
                continue

            is_marked = False
            for dec in node.decorators:
                expr = getattr(dec, "exper", None)
                if expr is None:
                    continue

                if RewriteWithConditionPass._decorator_kind(expr) is not None:
                    is_marked = True
                    break

            if not is_marked:
                continue

            sym_id = RewriteWithConditionPass._class_symbol_id(node, ctx)
            if sym_id is None or sym_id in classes:
                continue

            sym = ctx.symbols.get(SymbolId(sym_id))
            classes[sym_id] = _WithCondClassInfo(
                sym_id=sym_id,
                name=node.name,
                file=sym.decl_file if sym else ir.path,
                line=sym.decl_line if sym else node.line,
                offset=node.offset,
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

            module_name = ctx.module_name_by_path.get(info.file.resolve())
            if module_name is None:
                continue

            exports = ctx.module_exports.get(module_name) or {}
            for _name, sym_id in exports.items():
                export_owner[sym_id.value] = cls_id

    @staticmethod
    def _statement_lists(node: PyIRNode) -> Iterable[list[PyIRNode]]:
        if isinstance(node, PyIRFile):
            yield node.body

        elif isinstance(node, PyIRFunctionDef):
            yield node.body

        elif isinstance(node, PyIRClassDef):
            yield node.body

        elif isinstance(node, PyIRIf):
            yield node.body
            yield node.orelse

        elif isinstance(node, PyIRWhile):
            yield node.body

        elif isinstance(node, PyIRFor):
            yield node.body

        elif isinstance(node, PyIRWith):
            yield node.body

    @staticmethod
    def _err_ctx(file: Path | None, line: int | None, offset: int | None) -> str:
        return f"File: {file}\nLINE|OFFSET: {line}|{offset}"

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
            owner_cls = export_owner.get(expr.symbol_id)
            return owner_cls is not None and owner_cls in classes

        if isinstance(expr, PyIRVarUse):
            rid = RewriteWithConditionPass._symbol_id_of_root(expr, ctx)
            if rid is None:
                return False

            owner_cls = export_owner.get(rid)
            return owner_cls is not None and owner_cls in classes

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
            for child in RewriteWithConditionPass._statement_lists(st):
                child[:] = RewriteWithConditionPass._rewrite_stmt_list(
                    child, ctx=ctx, classes=classes, file=file
                )

            if isinstance(st, PyIRWith):
                repl = RewriteWithConditionPass._rewrite_with(st, classes, ctx, file)
                if repl is not None:
                    repl = RewriteWithConditionPass._rewrite_stmt_list(
                        repl, ctx=ctx, classes=classes, file=file
                    )
                    out.extend(repl)
                    continue

            out.append(st)

        return out

    @staticmethod
    def _rewrite_with(
        w: PyIRWith,
        classes: dict[int, _WithCondClassInfo],
        ctx: SymLinkContext,
        file: Path | None,
    ) -> list[PyIRNode] | None:
        cond_exprs: list[PyIRNode] = []

        for item in w.items:
            if item.optional_vars is not None:
                exit_with_code(
                    1,
                    "with_condition не поддерживает 'with ... as name'.\n"
                    + RewriteWithConditionPass._err_ctx(file, w.line, w.offset),
                )

            expr = item.context_expr

            if not RewriteWithConditionPass._is_with_condition_expr(
                expr, ctx=ctx, classes=classes
            ):
                return None

            cond_exprs.append(expr)

        if not cond_exprs:
            return None

        inner: list[PyIRNode] = list(w.body)
        with_scope = ctx.node_scope[ctx.uid(w)]

        for cond in reversed(cond_exprs):
            n_if = PyIRIf(
                line=w.line,
                offset=w.offset,
                test=cond,
                body=inner,
                orelse=[],
            )
            ctx.node_scope[ctx.uid(n_if)] = with_scope
            inner = [n_if]

        return inner

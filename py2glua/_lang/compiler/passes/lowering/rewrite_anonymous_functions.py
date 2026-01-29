from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRCall,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
    PyIRVarUse,
)
from ..analysis.symlinks import PyIRSymLink, SymbolId, SymLinkContext
from ..clone_utils import clone_deep
from ..macro_utils import MacroUtils

_DIRECTIVE_KIND: Literal["anonymous"] = "anonymous"


@dataclass
class PyIRFnExpr(PyIRNode):
    signature: dict[str, object]
    body: list[PyIRNode]

    def walk(self) -> Iterable[PyIRNode]:
        yield self
        for st in self.body:
            yield from st.walk()

    def __str__(self) -> str:
        return "<fn_expr>"


@dataclass
class _AnonFnInfo:
    sym_id: int
    name: str
    fn: PyIRFunctionDef
    file: Path | None
    line: int | None
    offset: int | None


class RewriteAnonymousFunctionsPass:
    """
    Lowering for @CompilerDirective.anonymous().

    Делает:
      - собирает anonymous-функции
      - заменяет каждое чтение символа функции на PyIRFnExpr
      - удаляет исходные def из AST
    """

    _CTX_SEEN = "_anon_seen_paths"
    _CTX_DONE = "_anon_done"
    _CTX_FUNCS = "_anon_funcs"  # sym_id -> _AnonFnInfo
    _CTX_FILES = "_anon_files"  # list[PyIRFile]

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        MacroUtils.ensure_pass_ctx(
            ctx,
            **{
                RewriteAnonymousFunctionsPass._CTX_SEEN: set(),
                RewriteAnonymousFunctionsPass._CTX_DONE: False,
                RewriteAnonymousFunctionsPass._CTX_FUNCS: {},
                RewriteAnonymousFunctionsPass._CTX_FILES: [],
            },
        )

        if ir.path is not None:
            getattr(ctx, RewriteAnonymousFunctionsPass._CTX_SEEN).add(ir.path.resolve())

        files: list[PyIRFile] = getattr(ctx, RewriteAnonymousFunctionsPass._CTX_FILES)
        if ir not in files:
            files.append(ir)

        RewriteAnonymousFunctionsPass._collect_anonymous_functions(ir, ctx)

        if MacroUtils.ready_to_finalize(
            ctx,
            seen_key=RewriteAnonymousFunctionsPass._CTX_SEEN,
            done_key=RewriteAnonymousFunctionsPass._CTX_DONE,
        ):
            funcs: dict[int, _AnonFnInfo] = getattr(
                ctx, RewriteAnonymousFunctionsPass._CTX_FUNCS
            )

            for file_ir in files:
                file_ir.body[:] = RewriteAnonymousFunctionsPass._rewrite_stmt_list(
                    file_ir.body,
                    ctx=ctx,
                    funcs=funcs,
                    file=file_ir.path,
                )

    @staticmethod
    def _func_symbol_id(fn: PyIRFunctionDef, ctx: SymLinkContext) -> int | None:
        if not fn.name:
            return None
        scope = ctx.node_scope[ctx.uid(fn)]
        link = ctx.resolve(scope=scope, name=fn.name)
        return None if link is None else link.target.value

    @staticmethod
    def _collect_anonymous_functions(ir: PyIRFile, ctx: SymLinkContext) -> None:
        funcs: dict[int, _AnonFnInfo] = getattr(
            ctx, RewriteAnonymousFunctionsPass._CTX_FUNCS
        )

        for node in ir.walk():
            if not isinstance(node, PyIRFunctionDef):
                continue
            if not node.name:
                continue

            kinds: set[str] = set()
            for dec in node.decorators:
                expr = getattr(dec, "exper", None)
                if expr and MacroUtils.decorator_kind(
                    expr, ctx=ctx, kind=_DIRECTIVE_KIND
                ):
                    kinds.add(_DIRECTIVE_KIND)

            if not kinds:
                continue

            sym_id = RewriteAnonymousFunctionsPass._func_symbol_id(node, ctx)
            if sym_id is None or sym_id in funcs:
                continue

            sym = ctx.symbols.get(SymbolId(sym_id))
            funcs[sym_id] = _AnonFnInfo(
                sym_id=sym_id,
                name=node.name,
                fn=node,
                file=sym.decl_file if sym else ir.path,
                line=sym.decl_line if sym else node.line,
                offset=node.offset,
            )

    @staticmethod
    def _make_fn_expr(
        info: _AnonFnInfo,
        *,
        line: int | None,
        offset: int | None,
    ) -> PyIRFnExpr:
        return PyIRFnExpr(
            line=line,
            offset=offset,
            signature=dict(info.fn.signature),
            body=[clone_deep(st) for st in info.fn.body],
        )

    @staticmethod
    def _forbidden_anon_call(
        *,
        call: PyIRCall,
        info: _AnonFnInfo,
        file: Path | None,
    ) -> None:
        CompilerExit.user_error_node(
            "Вызов функции, помеченной @CompilerDirective.anonymous(), запрещён.\n"
            f"Функция: {info.name}",
            file,
            call,
        )

    @staticmethod
    def _rewrite_stmt_list(
        stmts: list[PyIRNode],
        *,
        ctx: SymLinkContext,
        funcs: dict[int, _AnonFnInfo],
        file: Path | None,
    ) -> list[PyIRNode]:
        out: list[PyIRNode] = []

        for st in stmts:
            for child in MacroUtils.statement_lists(st):
                child[:] = RewriteAnonymousFunctionsPass._rewrite_stmt_list(
                    child, ctx=ctx, funcs=funcs, file=file
                )

            st = RewriteAnonymousFunctionsPass._rw_node(
                st, ctx=ctx, funcs=funcs, store=False, file=file
            )

            if isinstance(st, PyIRFunctionDef) and st.name:
                sym_id = RewriteAnonymousFunctionsPass._func_symbol_id(st, ctx)
                if sym_id is not None and sym_id in funcs:
                    continue

            out.append(st)

        return out

    @staticmethod
    def _rw_node(
        node: PyIRNode,
        *,
        ctx: SymLinkContext,
        funcs: dict[int, _AnonFnInfo],
        store: bool,
        file: Path | None,
    ) -> PyIRNode:
        if isinstance(node, PyIRCall):
            f = node.func

            if isinstance(f, PyIRSymLink) and not f.is_store:
                info = funcs.get(f.symbol_id)
                if info:
                    RewriteAnonymousFunctionsPass._forbidden_anon_call(
                        call=node, info=info, file=file
                    )

            if isinstance(f, PyIRVarUse):
                link = ctx.var_links.get(ctx.uid(f))
                if link:
                    info = funcs.get(link.target.value)
                    if info:
                        RewriteAnonymousFunctionsPass._forbidden_anon_call(
                            call=node, info=info, file=file
                        )

            node.func = RewriteAnonymousFunctionsPass._rw_node(
                node.func, ctx=ctx, funcs=funcs, store=False, file=file
            )
            node.args_p = [
                RewriteAnonymousFunctionsPass._rw_node(
                    a, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for a in node.args_p
            ]
            node.args_kw = {
                k: RewriteAnonymousFunctionsPass._rw_node(
                    v, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for k, v in node.args_kw.items()
            }
            return node

        if isinstance(node, PyIRSymLink):
            if store or node.is_store:
                return node

            info = funcs.get(node.symbol_id)
            if not info:
                return node

            return RewriteAnonymousFunctionsPass._make_fn_expr(
                info, line=node.line, offset=node.offset
            )

        if isinstance(node, PyIRVarUse):
            if store:
                return node

            link = ctx.var_links.get(ctx.uid(node))
            if not link:
                return node

            info = funcs.get(link.target.value)
            if not info:
                return node

            return RewriteAnonymousFunctionsPass._make_fn_expr(
                info, line=node.line, offset=node.offset
            )

        for child_lists in MacroUtils.statement_lists(node):
            child_lists[:] = [
                RewriteAnonymousFunctionsPass._rw_node(
                    n, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for n in child_lists
            ]

        return node

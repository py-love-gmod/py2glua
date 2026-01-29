from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRCall,
    PyIRConstant,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
    PyIRVarUse,
    PyIRWith,
)
from ..analysis.symlinks import PyIRSymLink, SymbolId, SymLinkContext
from ..clone_utils import clone_with_subst
from ..macro_utils import MacroUtils

_DIRECTIVE_KIND: Literal["contextmanager"] = "contextmanager"


@dataclass
class _CtxmFnInfo:
    sym_id: int
    name: str
    fn: PyIRFunctionDef
    file: Path | None
    line: int | None
    offset: int | None


class RewriteContextManagerWithPass:
    """
    Макро-развёртка with для @CompilerDirective.contextmanager().

    Стековая семантика multiple-items:
        with A(...), B(...):
            body
    ->
        A.pre(...)
        B.pre(...)
        body
        B.post(...)
        A.post(...)

    Правила:
      - marker contextmanager_body <= 1
      - marker == 0 -> тело with игнорируется
    """

    _CTX_SEEN = "_ctxm_seen_paths"
    _CTX_DONE = "_ctxm_done"
    _CTX_FUNCS = "_ctxm_funcs"  # sym_id -> _CtxmFnInfo
    _CTX_TMP_COUNTER = "_ctxm_tmp_counter"
    _CTX_FILES = "_ctxm_files"

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        MacroUtils.ensure_pass_ctx(
            ctx,
            **{
                RewriteContextManagerWithPass._CTX_SEEN: set(),
                RewriteContextManagerWithPass._CTX_DONE: False,
                RewriteContextManagerWithPass._CTX_FUNCS: {},
                RewriteContextManagerWithPass._CTX_TMP_COUNTER: 0,
                RewriteContextManagerWithPass._CTX_FILES: [],
            },
        )

        if ir.path is not None:
            getattr(ctx, RewriteContextManagerWithPass._CTX_SEEN).add(ir.path.resolve())

        files: list[PyIRFile] = getattr(ctx, RewriteContextManagerWithPass._CTX_FILES)
        if ir not in files:
            files.append(ir)

        RewriteContextManagerWithPass._collect_contextmanager_functions(ir, ctx)

        if MacroUtils.ready_to_finalize(
            ctx,
            seen_key=RewriteContextManagerWithPass._CTX_SEEN,
            done_key=RewriteContextManagerWithPass._CTX_DONE,
        ):
            funcs: dict[int, _CtxmFnInfo] = getattr(
                ctx, RewriteContextManagerWithPass._CTX_FUNCS
            )

            for file_ir in files:
                file_ir.body[:] = RewriteContextManagerWithPass._rewrite_stmt_list(
                    file_ir.body,
                    ctx=ctx,
                    funcs=funcs,
                    file=file_ir.path,
                )

    @staticmethod
    def _func_symbol_id(fn: PyIRFunctionDef, ctx: SymLinkContext) -> int | None:
        scope = ctx.node_scope[ctx.uid(fn)]
        link = ctx.resolve(scope=scope, name=fn.name)
        return None if link is None else link.target.value

    @staticmethod
    def _collect_contextmanager_functions(ir: PyIRFile, ctx: SymLinkContext) -> None:
        funcs: dict[int, _CtxmFnInfo] = getattr(
            ctx, RewriteContextManagerWithPass._CTX_FUNCS
        )

        for fn in ir.walk():
            if not isinstance(fn, PyIRFunctionDef):
                continue

            found = False
            for dec in fn.decorators:
                expr = getattr(dec, "exper", None)
                if expr and MacroUtils.decorator_kind(
                    expr, ctx=ctx, kind=_DIRECTIVE_KIND
                ):
                    found = True
                    break

            if not found:
                continue

            sym_id = RewriteContextManagerWithPass._func_symbol_id(fn, ctx)
            if sym_id is None or sym_id in funcs:
                continue

            sym = ctx.symbols.get(SymbolId(sym_id))
            funcs[sym_id] = _CtxmFnInfo(
                sym_id=sym_id,
                name=fn.name,
                fn=fn,
                file=sym.decl_file if sym else ir.path,
                line=sym.decl_line if sym else fn.line,
                offset=fn.offset,
            )

    @staticmethod
    def _rewrite_stmt_list(
        stmts: list[PyIRNode],
        *,
        ctx: SymLinkContext,
        funcs: dict[int, _CtxmFnInfo],
        file: Path | None,
    ) -> list[PyIRNode]:
        out: list[PyIRNode] = []

        for st in stmts:
            for child in MacroUtils.statement_lists(st):
                child[:] = RewriteContextManagerWithPass._rewrite_stmt_list(
                    child, ctx=ctx, funcs=funcs, file=file
                )

            if isinstance(st, PyIRWith):
                repl = RewriteContextManagerWithPass._rewrite_with(st, funcs, ctx, file)
                if repl is not None:
                    repl = RewriteContextManagerWithPass._rewrite_stmt_list(
                        repl, ctx=ctx, funcs=funcs, file=file
                    )
                    out.extend(repl)
                    continue

            out.append(st)

        return out

    @staticmethod
    def _is_ctx_body_marker(node: PyIRNode) -> bool:
        return (
            (isinstance(node, PyIRAttribute) and node.attr == "contextmanager_body")
            or (isinstance(node, PyIRVarUse) and node.name == "contextmanager_body")
            or (isinstance(node, PyIRSymLink) and node.name == "contextmanager_body")
        )

    @staticmethod
    def _count_markers(body: list[PyIRNode]) -> int:
        cnt = 0
        stack: list[list[PyIRNode]] = [body]
        while stack:
            lst = stack.pop()
            for st in lst:
                if RewriteContextManagerWithPass._is_ctx_body_marker(st):
                    cnt += 1

                for child in MacroUtils.statement_lists(st):
                    stack.append(child)

        return cnt

    @staticmethod
    def _find_marker(body: list[PyIRNode]) -> tuple[list[PyIRNode], int] | None:
        stack: list[list[PyIRNode]] = [body]
        while stack:
            lst = stack.pop()
            for i, st in enumerate(lst):
                if RewriteContextManagerWithPass._is_ctx_body_marker(st):
                    return lst, i

                for child in MacroUtils.statement_lists(st):
                    stack.append(child)

        return None

    @staticmethod
    def _next_tmp_name(ctx: SymLinkContext, scope, *, prefix: str = "__ctxm_") -> str:
        while True:
            cur = getattr(ctx, RewriteContextManagerWithPass._CTX_TMP_COUNTER)
            setattr(ctx, RewriteContextManagerWithPass._CTX_TMP_COUNTER, cur + 1)
            name = f"{prefix}{cur}"
            if name not in ctx.scopes[scope].defs:
                return name

    @staticmethod
    def _is_pure_expr(node: PyIRNode) -> bool:
        return isinstance(node, (PyIRVarUse, PyIRSymLink, PyIRConstant))

    @staticmethod
    def _expand_macro_call(
        *,
        w: PyIRWith,
        with_scope,
        macro: _CtxmFnInfo,
        call: PyIRCall,
        inner_body: list[PyIRNode],
        ctx: SymLinkContext,
        file: Path | None,
    ) -> list[PyIRNode]:
        fn = macro.fn
        param_names = list(fn.signature.keys())
        bound_pairs = list(zip(param_names, call.args_p))

        eval_nodes: list[PyIRNode] = []
        subst: dict[str, PyIRNode] = {}

        for pname, arg in bound_pairs:
            if RewriteContextManagerWithPass._is_pure_expr(arg):
                subst[pname] = clone_with_subst(arg, {})
                continue

            tmp = RewriteContextManagerWithPass._next_tmp_name(ctx, with_scope)
            sym_id = ctx.ensure_def(
                scope=with_scope,
                name=tmp,
                decl_file=file,
                decl_line=w.line,
            )
            sym = ctx.symbols[sym_id]

            store = PyIRSymLink(
                line=w.line,
                offset=w.offset,
                name=tmp,
                symbol_id=sym.id.value,
                hops=0,
                is_store=True,
                decl_file=sym.decl_file,
                decl_line=sym.decl_line,
            )
            load = PyIRSymLink(
                line=w.line,
                offset=w.offset,
                name=tmp,
                symbol_id=sym.id.value,
                hops=0,
                is_store=False,
                decl_file=sym.decl_file,
                decl_line=sym.decl_line,
            )

            ctx.node_scope[ctx.uid(store)] = with_scope
            ctx.node_scope[ctx.uid(load)] = with_scope

            eval_nodes.append(
                PyIRAssign(
                    line=w.line,
                    offset=w.offset,
                    targets=[store],
                    value=clone_with_subst(arg, {}),
                )
            )
            subst[pname] = load

        macro_body = [clone_with_subst(n, subst) for n in fn.body]

        cnt = RewriteContextManagerWithPass._count_markers(macro_body)
        if cnt > 1:
            CompilerExit.user_error(
                "Маркер contextmanager_body может встречаться не более одного раза.\n"
                f"Макрос: {macro.name}",
                macro.file,
                macro.line,
                macro.offset,
            )

        hit = RewriteContextManagerWithPass._find_marker(macro_body)
        if hit:
            parent, idx = hit
            parent[idx : idx + 1] = inner_body

        return [*eval_nodes, *macro_body]

    @staticmethod
    def _rewrite_with(
        w: PyIRWith,
        funcs: dict[int, _CtxmFnInfo],
        ctx: SymLinkContext,
        file: Path | None,
    ) -> list[PyIRNode] | None:
        calls: list[tuple[PyIRCall, _CtxmFnInfo]] = []

        for item in w.items:
            if not isinstance(item.context_expr, PyIRCall):
                return None

            call = item.context_expr
            if not isinstance(call.func, PyIRSymLink):
                return None

            info = funcs.get(call.func.symbol_id)
            if info is None:
                return None

            calls.append((call, info))

        if not calls:
            return None

        with_scope = ctx.node_scope[ctx.uid(w)]
        cur = list(w.body)

        for call, info in reversed(calls):
            cur = RewriteContextManagerWithPass._expand_macro_call(
                w=w,
                with_scope=with_scope,
                macro=info,
                call=call,
                inner_body=cur,
                ctx=ctx,
                file=file,
            )

        return cur

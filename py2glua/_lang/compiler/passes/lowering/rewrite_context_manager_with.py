from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Iterable, Literal

from ....._cli.logging_setup import exit_with_code
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
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

    Механика:
      - берём тело макро-функции
      - подставляем args в параметры
      - находим statement marker `contextmanager_body`
      - заменяем marker на тело with

    Правила:
      - marker может быть в любом statement-list.
      - marker > 1 -> ошибка.
      - marker == 0 -> тело with игнорируется.
      - "with ... as name" запрещён.
      - kwargs/кол-во аргументов тут не валидируем.
    """

    _CTX_SEEN = "_ctxm_seen_paths"
    _CTX_DONE = "_ctxm_done"
    _CTX_FUNCS = "_ctxm_funcs"  # sym_id -> _CtxmFnInfo
    _CTX_TMP_COUNTER = "_ctxm_tmp_counter"
    _CTX_FILES = "_ctxm_files"  # list[PyIRFile]

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        RewriteContextManagerWithPass._ensure_ctx(ctx)

        if ir.path is not None:
            getattr(ctx, RewriteContextManagerWithPass._CTX_SEEN).add(ir.path.resolve())

        files: list[PyIRFile] = getattr(ctx, RewriteContextManagerWithPass._CTX_FILES)
        if ir not in files:
            files.append(ir)

        RewriteContextManagerWithPass._collect_contextmanager_functions(ir, ctx)
        RewriteContextManagerWithPass._maybe_finalize(ctx)

    @staticmethod
    def _ensure_ctx(ctx: SymLinkContext) -> None:
        if not hasattr(ctx, RewriteContextManagerWithPass._CTX_SEEN):
            setattr(ctx, RewriteContextManagerWithPass._CTX_SEEN, set())

        if not hasattr(ctx, RewriteContextManagerWithPass._CTX_DONE):
            setattr(ctx, RewriteContextManagerWithPass._CTX_DONE, False)

        if not hasattr(ctx, RewriteContextManagerWithPass._CTX_FUNCS):
            setattr(ctx, RewriteContextManagerWithPass._CTX_FUNCS, {})

        if not hasattr(ctx, RewriteContextManagerWithPass._CTX_TMP_COUNTER):
            setattr(ctx, RewriteContextManagerWithPass._CTX_TMP_COUNTER, 0)

        if not hasattr(ctx, RewriteContextManagerWithPass._CTX_FILES):
            setattr(ctx, RewriteContextManagerWithPass._CTX_FILES, [])

    @staticmethod
    def _maybe_finalize(ctx: SymLinkContext) -> None:
        if getattr(ctx, RewriteContextManagerWithPass._CTX_DONE):
            return

        all_paths = {p.resolve() for p in getattr(ctx, "_all_paths", set()) if p}
        seen = getattr(ctx, RewriteContextManagerWithPass._CTX_SEEN)
        if all_paths and seen != all_paths:
            return

        setattr(ctx, RewriteContextManagerWithPass._CTX_DONE, True)

        funcs: dict[int, _CtxmFnInfo] = getattr(
            ctx, RewriteContextManagerWithPass._CTX_FUNCS
        )
        files: list[PyIRFile] = getattr(ctx, RewriteContextManagerWithPass._CTX_FILES)

        for ir in files:
            ir.body[:] = RewriteContextManagerWithPass._rewrite_stmt_list(
                ir.body, ctx=ctx, funcs=funcs, file=ir.path
            )

    @staticmethod
    def _decorator_kind(expr: PyIRNode) -> Literal["contextmanager"] | None:
        if isinstance(expr, PyIRAttribute) and expr.attr == "contextmanager":
            return expr.attr  # type: ignore[return-value]

        if isinstance(expr, PyIRCall):
            f = expr.func
            if isinstance(f, PyIRAttribute) and f.attr == "contextmanager":
                return f.attr  # type: ignore[return-value]

            if isinstance(f, PyIRSymLink) and f.name == "contextmanager":
                return f.name  # type: ignore[return-value]

        return None

    @staticmethod
    def _func_symbol_id(fn: PyIRFunctionDef, ctx: SymLinkContext) -> int | None:
        decl_scope = ctx.node_scope[ctx.uid(fn)]
        link = ctx.resolve(scope=decl_scope, name=fn.name)
        return None if link is None else link.target.value

    @staticmethod
    def _collect_contextmanager_functions(ir: PyIRFile, ctx: SymLinkContext) -> None:
        funcs: dict[int, _CtxmFnInfo] = getattr(
            ctx, RewriteContextManagerWithPass._CTX_FUNCS
        )

        for node in ir.walk():
            if not isinstance(node, PyIRFunctionDef):
                continue

            is_ctxm = False
            for dec in node.decorators:
                expr = getattr(dec, "exper", None)
                if expr is None:
                    continue

                if RewriteContextManagerWithPass._decorator_kind(expr) is not None:
                    is_ctxm = True

            if not is_ctxm:
                continue

            sym_id = RewriteContextManagerWithPass._func_symbol_id(node, ctx)
            if sym_id is None or sym_id in funcs:
                continue

            sym = ctx.symbols.get(SymbolId(sym_id))
            funcs[sym_id] = _CtxmFnInfo(
                sym_id=sym_id,
                name=node.name,
                fn=node,
                file=sym.decl_file if sym else ir.path,
                line=sym.decl_line if sym else node.line,
                offset=node.offset,
            )

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
    def _rewrite_stmt_list(
        stmts: list[PyIRNode],
        *,
        ctx: SymLinkContext,
        funcs: dict[int, _CtxmFnInfo],
        file: Path | None,
    ) -> list[PyIRNode]:
        out: list[PyIRNode] = []

        for st in stmts:
            for child in RewriteContextManagerWithPass._statement_lists(st):
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
    def _count_markers_in_stmt_lists(body: list[PyIRNode]) -> int:
        cnt = 0
        stack: list[list[PyIRNode]] = [body]
        while stack:
            lst = stack.pop()
            for st in lst:
                if RewriteContextManagerWithPass._is_ctx_body_marker(st):
                    cnt += 1

                for child in RewriteContextManagerWithPass._statement_lists(st):
                    stack.append(child)

        return cnt

    @staticmethod
    def _find_marker_stmt(body: list[PyIRNode]) -> tuple[list[PyIRNode], int] | None:
        stack: list[list[PyIRNode]] = [body]
        while stack:
            lst = stack.pop()
            for i, st in enumerate(lst):
                if RewriteContextManagerWithPass._is_ctx_body_marker(st):
                    return lst, i

                for child in RewriteContextManagerWithPass._statement_lists(st):
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
    def _clone_with_subst(node: PyIRNode, subst: dict[str, PyIRNode]) -> PyIRNode:
        def clone_value(v, *, apply_subst: bool):
            if isinstance(v, PyIRNode):
                return clone_node(v, apply_subst=apply_subst)

            if isinstance(v, list):
                return [clone_value(x, apply_subst=apply_subst) for x in v]

            if isinstance(v, dict):
                return {
                    k: clone_value(x, apply_subst=apply_subst) for k, x in v.items()
                }

            return v

        def force_load(n: PyIRNode) -> PyIRNode:
            if isinstance(n, PyIRSymLink) and n.is_store:
                return PyIRSymLink(
                    line=n.line,
                    offset=n.offset,
                    name=n.name,
                    symbol_id=n.symbol_id,
                    hops=n.hops,
                    is_store=False,
                    decl_file=n.decl_file,
                    decl_line=n.decl_line,
                )
            return n

        def clone_node(n: PyIRNode, *, apply_subst: bool) -> PyIRNode:
            if apply_subst:
                if isinstance(n, PyIRVarUse) and n.name in subst:
                    return clone_node(subst[n.name], apply_subst=False)

                if isinstance(n, PyIRSymLink) and n.name in subst:
                    return clone_node(subst[n.name], apply_subst=False)

            if not is_dataclass(n):
                return n

            cls = n.__class__
            kwargs = {}
            for f in fields(n):
                value = getattr(n, f.name)
                if f.name in ("line", "offset"):
                    kwargs[f.name] = value

                else:
                    kwargs[f.name] = clone_value(value, apply_subst=apply_subst)

            out = cls(**kwargs)
            if isinstance(out, PyIRSymLink):
                out = force_load(out)
            return out

        return clone_node(node, apply_subst=True)

    @staticmethod
    def _err_ctx(file: Path | None, line: int | None, offset: int | None) -> str:
        return f"Файл: {file}\nLINE|OFFSET: {line}|{offset}"

    @staticmethod
    def _define_temp_symlink(
        ctx: SymLinkContext,
        *,
        scope,
        name: str,
        file: Path | None,
        line: int | None,
        offset: int | None,
    ) -> tuple[PyIRSymLink, PyIRSymLink]:
        sym_id = ctx.ensure_def(scope=scope, name=name, decl_file=file, decl_line=line)
        sym = ctx.symbols[sym_id]

        store = PyIRSymLink(
            line=line,
            offset=offset,
            name=name,
            symbol_id=sym.id.value,
            hops=0,
            is_store=True,
            decl_file=sym.decl_file,
            decl_line=sym.decl_line,
        )
        load = PyIRSymLink(
            line=line,
            offset=offset,
            name=name,
            symbol_id=sym.id.value,
            hops=0,
            is_store=False,
            decl_file=sym.decl_file,
            decl_line=sym.decl_line,
        )

        ctx.node_scope[ctx.uid(store)] = scope
        ctx.node_scope[ctx.uid(load)] = scope
        return store, load

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

        for pname, arg_node in bound_pairs:
            if RewriteContextManagerWithPass._is_pure_expr(arg_node):
                subst[pname] = RewriteContextManagerWithPass._clone_with_subst(
                    arg_node, {}
                )
                continue

            tmp_name = RewriteContextManagerWithPass._next_tmp_name(ctx, with_scope)
            store_sym, load_sym = RewriteContextManagerWithPass._define_temp_symlink(
                ctx,
                scope=with_scope,
                name=tmp_name,
                file=file,
                line=w.line,
                offset=w.offset,
            )

            eval_nodes.append(
                PyIRAssign(
                    line=w.line,
                    offset=w.offset,
                    targets=[store_sym],
                    value=RewriteContextManagerWithPass._clone_with_subst(arg_node, {}),
                )
            )
            subst[pname] = load_sym

        macro_body = [
            RewriteContextManagerWithPass._clone_with_subst(n, subst) for n in fn.body
        ]

        cnt = RewriteContextManagerWithPass._count_markers_in_stmt_lists(macro_body)
        if cnt > 1:
            exit_with_code(
                1,
                "Маркер contextmanager_body может встречаться не более одного раза.\n"
                f"Макрос: {macro.name}\n"
                + RewriteContextManagerWithPass._err_ctx(
                    macro.file, macro.line, macro.offset
                ),
            )

        hit = RewriteContextManagerWithPass._find_marker_stmt(macro_body)
        if hit:
            parent_list, idx = hit
            parent_list[idx : idx + 1] = inner_body

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
            if item.optional_vars is not None:
                exit_with_code(
                    1,
                    "Макро-contextmanager не поддерживает конструкцию 'with ... as имя'.\n"
                    + RewriteContextManagerWithPass._err_ctx(file, w.line, w.offset),
                )

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
        cur: list[PyIRNode] = list(w.body)
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

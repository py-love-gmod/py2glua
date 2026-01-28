from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Iterable, Literal

from ....._cli.logging_setup import exit_with_code
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
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
    PyIRWithItem,
)
from ..analysis.symlinks import PyIRSymLink, SymbolId, SymLinkContext


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
        RewriteAnonymousFunctionsPass._ensure_ctx(ctx)

        if ir.path is not None:
            getattr(ctx, RewriteAnonymousFunctionsPass._CTX_SEEN).add(ir.path.resolve())

        files: list[PyIRFile] = getattr(ctx, RewriteAnonymousFunctionsPass._CTX_FILES)
        if ir not in files:
            files.append(ir)

        RewriteAnonymousFunctionsPass._collect_anonymous_functions(ir, ctx)
        RewriteAnonymousFunctionsPass._maybe_finalize(ctx)

    @staticmethod
    def _ensure_ctx(ctx: SymLinkContext) -> None:
        if not hasattr(ctx, RewriteAnonymousFunctionsPass._CTX_SEEN):
            setattr(ctx, RewriteAnonymousFunctionsPass._CTX_SEEN, set())

        if not hasattr(ctx, RewriteAnonymousFunctionsPass._CTX_DONE):
            setattr(ctx, RewriteAnonymousFunctionsPass._CTX_DONE, False)

        if not hasattr(ctx, RewriteAnonymousFunctionsPass._CTX_FUNCS):
            setattr(ctx, RewriteAnonymousFunctionsPass._CTX_FUNCS, {})

        if not hasattr(ctx, RewriteAnonymousFunctionsPass._CTX_FILES):
            setattr(ctx, RewriteAnonymousFunctionsPass._CTX_FILES, [])

    @staticmethod
    def _maybe_finalize(ctx: SymLinkContext) -> None:
        if getattr(ctx, RewriteAnonymousFunctionsPass._CTX_DONE):
            return

        all_paths = {p.resolve() for p in getattr(ctx, "_all_paths", set()) if p}
        seen = getattr(ctx, RewriteAnonymousFunctionsPass._CTX_SEEN)
        if all_paths and seen != all_paths:
            return

        setattr(ctx, RewriteAnonymousFunctionsPass._CTX_DONE, True)

        funcs: dict[int, _AnonFnInfo] = getattr(
            ctx, RewriteAnonymousFunctionsPass._CTX_FUNCS
        )
        files: list[PyIRFile] = getattr(ctx, RewriteAnonymousFunctionsPass._CTX_FILES)

        for ir in files:
            ir.body[:] = RewriteAnonymousFunctionsPass._rewrite_stmt_list(
                ir.body, ctx=ctx, funcs=funcs, file=ir.path
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
    def _decorator_expr(dec: PyIRNode) -> PyIRNode | None:
        return getattr(dec, "exper", None) or getattr(dec, "expr", None)

    @staticmethod
    def _decorator_kind(expr: PyIRNode) -> Literal["anonymous"] | None:
        if isinstance(expr, PyIRAttribute) and expr.attr == "anonymous":
            return "anonymous"

        if isinstance(expr, PyIRCall):
            f = expr.func
            if isinstance(f, PyIRAttribute) and f.attr == "anonymous":
                return "anonymous"

            if isinstance(f, PyIRSymLink) and f.name == "anonymous":
                return "anonymous"

        if isinstance(expr, PyIRSymLink) and expr.name == "anonymous":
            return "anonymous"

        return None

    @staticmethod
    def _is_anonymous_fn(fn: PyIRFunctionDef) -> bool:
        for dec in fn.decorators:
            expr = RewriteAnonymousFunctionsPass._decorator_expr(dec)
            if expr is None:
                continue

            if RewriteAnonymousFunctionsPass._decorator_kind(expr) is not None:
                return True

        return False

    @staticmethod
    def _func_symbol_id(fn: PyIRFunctionDef, ctx: SymLinkContext) -> int | None:
        if not fn.name:
            return None

        decl_scope = ctx.node_scope[ctx.uid(fn)]
        link = ctx.resolve(scope=decl_scope, name=fn.name)
        return None if link is None else link.target.value

    @staticmethod
    def _err_ctx(file: Path | None, line: int | None, offset: int | None) -> str:
        return f"Файл: {file}\nLINE|OFFSET: {line}|{offset}"

    @staticmethod
    def _collect_anonymous_functions(ir: PyIRFile, ctx: SymLinkContext) -> None:
        funcs: dict[int, _AnonFnInfo] = getattr(
            ctx, RewriteAnonymousFunctionsPass._CTX_FUNCS
        )

        stack: list[list[PyIRNode]] = [ir.body]
        while stack:
            lst = stack.pop()
            for st in lst:
                for child in RewriteAnonymousFunctionsPass._statement_lists(st):
                    stack.append(child)

                if not isinstance(st, PyIRFunctionDef):
                    continue

                if not st.name:
                    continue

                if not RewriteAnonymousFunctionsPass._is_anonymous_fn(st):
                    continue

                sym_id = RewriteAnonymousFunctionsPass._func_symbol_id(st, ctx)
                if sym_id is None or sym_id in funcs:
                    continue

                sym = ctx.symbols.get(SymbolId(sym_id))
                funcs[sym_id] = _AnonFnInfo(
                    sym_id=sym_id,
                    name=st.name,
                    fn=st,
                    file=sym.decl_file if sym else ir.path,
                    line=sym.decl_line if sym else st.line,
                    offset=st.offset,
                )

    @staticmethod
    def _clone_node(v):
        if isinstance(v, PyIRNode):
            return RewriteAnonymousFunctionsPass._clone_dataclass_node(v)

        if isinstance(v, list):
            return [RewriteAnonymousFunctionsPass._clone_node(x) for x in v]

        if isinstance(v, dict):
            return {
                k: RewriteAnonymousFunctionsPass._clone_node(x) for k, x in v.items()
            }

        return v

    @staticmethod
    def _clone_dataclass_node(n: PyIRNode) -> PyIRNode:
        if not is_dataclass(n):
            return n

        cls = n.__class__
        kwargs: dict[str, object] = {}
        for f in fields(n):
            kwargs[f.name] = RewriteAnonymousFunctionsPass._clone_node(
                getattr(n, f.name)
            )

        return cls(**kwargs)  # type: ignore[misc]

    @staticmethod
    def _make_fn_expr(
        info: _AnonFnInfo, *, line: int | None, offset: int | None
    ) -> PyIRFnExpr:
        body = [
            RewriteAnonymousFunctionsPass._clone_dataclass_node(st)
            for st in info.fn.body
        ]
        sig = dict(info.fn.signature)
        return PyIRFnExpr(line=line, offset=offset, signature=sig, body=body)  # type: ignore[call-arg]

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
            for child in RewriteAnonymousFunctionsPass._statement_lists(st):
                child[:] = RewriteAnonymousFunctionsPass._rewrite_stmt_list(
                    child, ctx=ctx, funcs=funcs, file=file
                )

            st = RewriteAnonymousFunctionsPass._rw_node(
                st, ctx=ctx, funcs=funcs, store=False, file=file
            )

            if isinstance(st, PyIRFunctionDef) and st.name:
                sym_id = RewriteAnonymousFunctionsPass._func_symbol_id(st, ctx)
                if (
                    sym_id is not None
                    and sym_id in funcs
                    and RewriteAnonymousFunctionsPass._is_anonymous_fn(st)
                ):
                    continue

            out.append(st)

        return out

    @staticmethod
    def _forbidden_anon_call(
        *,
        call: PyIRCall,
        info: _AnonFnInfo,
        file: Path | None,
    ) -> None:
        exit_with_code(
            1,
            "Вызов функции, помеченной @CompilerDirective.anonymous(), запрещён.\n"
            f"Функция: {info.name}\n"
            + RewriteAnonymousFunctionsPass._err_ctx(file, call.line, call.offset),
        )

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

            if isinstance(f, PyIRSymLink) and (not f.is_store):
                info = funcs.get(f.symbol_id)
                if info is not None:
                    RewriteAnonymousFunctionsPass._forbidden_anon_call(
                        call=node, info=info, file=file
                    )

            if isinstance(f, PyIRVarUse):
                uid = ctx.uid(f)
                link = ctx.var_links.get(uid)
                if link is not None:
                    info = funcs.get(link.target.value)
                    if info is not None:
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
            if info is None:
                return node

            return RewriteAnonymousFunctionsPass._make_fn_expr(
                info, line=node.line, offset=node.offset
            )

        if isinstance(node, PyIRVarUse):
            if store:
                return node

            uid = ctx.uid(node)
            link = ctx.var_links.get(uid)
            if link is None:
                return node

            info = funcs.get(link.target.value)
            if info is None:
                return node

            return RewriteAnonymousFunctionsPass._make_fn_expr(
                info, line=node.line, offset=node.offset
            )

        if isinstance(node, PyIRAttribute):
            node.value = RewriteAnonymousFunctionsPass._rw_node(
                node.value, ctx=ctx, funcs=funcs, store=False, file=file
            )
            return node

        if isinstance(node, PyIRAssign):
            node.targets = [
                RewriteAnonymousFunctionsPass._rw_node(
                    t, ctx=ctx, funcs=funcs, store=True, file=file
                )
                for t in node.targets
            ]
            node.value = RewriteAnonymousFunctionsPass._rw_node(
                node.value, ctx=ctx, funcs=funcs, store=False, file=file
            )
            return node

        if isinstance(node, PyIRAugAssign):
            node.target = RewriteAnonymousFunctionsPass._rw_node(
                node.target, ctx=ctx, funcs=funcs, store=True, file=file
            )
            node.value = RewriteAnonymousFunctionsPass._rw_node(
                node.value, ctx=ctx, funcs=funcs, store=False, file=file
            )
            return node

        if isinstance(node, PyIRFor):
            node.target = RewriteAnonymousFunctionsPass._rw_node(
                node.target, ctx=ctx, funcs=funcs, store=True, file=file
            )
            node.iter = RewriteAnonymousFunctionsPass._rw_node(
                node.iter, ctx=ctx, funcs=funcs, store=False, file=file
            )
            node.body = [
                RewriteAnonymousFunctionsPass._rw_node(
                    n, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for n in node.body
            ]
            return node

        if isinstance(node, PyIRWithItem):
            node.context_expr = RewriteAnonymousFunctionsPass._rw_node(
                node.context_expr, ctx=ctx, funcs=funcs, store=False, file=file
            )
            if node.optional_vars is not None:
                node.optional_vars = RewriteAnonymousFunctionsPass._rw_node(
                    node.optional_vars, ctx=ctx, funcs=funcs, store=True, file=file
                )
            return node

        if isinstance(node, PyIRIf):
            node.test = RewriteAnonymousFunctionsPass._rw_node(
                node.test, ctx=ctx, funcs=funcs, store=False, file=file
            )
            node.body = [
                RewriteAnonymousFunctionsPass._rw_node(
                    n, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for n in node.body
            ]
            node.orelse = [
                RewriteAnonymousFunctionsPass._rw_node(
                    n, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for n in node.orelse
            ]
            return node

        if isinstance(node, PyIRWhile):
            node.test = RewriteAnonymousFunctionsPass._rw_node(
                node.test, ctx=ctx, funcs=funcs, store=False, file=file
            )
            node.body = [
                RewriteAnonymousFunctionsPass._rw_node(
                    n, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for n in node.body
            ]
            return node

        if isinstance(node, PyIRFunctionDef):
            node.decorators = [  # type: ignore[attr-defined]
                RewriteAnonymousFunctionsPass._rw_node(
                    d, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for d in node.decorators
            ]
            node.body = [
                RewriteAnonymousFunctionsPass._rw_node(
                    n, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for n in node.body
            ]
            return node

        if isinstance(node, PyIRClassDef):
            node.decorators = [  # type: ignore[attr-defined]
                RewriteAnonymousFunctionsPass._rw_node(
                    d, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for d in node.decorators
            ]
            node.bases = [
                RewriteAnonymousFunctionsPass._rw_node(
                    b, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for b in node.bases
            ]
            node.body = [
                RewriteAnonymousFunctionsPass._rw_node(
                    n, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for n in node.body
            ]
            return node

        if isinstance(node, PyIRFnExpr):
            node.body = [
                RewriteAnonymousFunctionsPass._rw_node(
                    n, ctx=ctx, funcs=funcs, store=False, file=file
                )
                for n in node.body
            ]
            return node

        if is_dataclass(node):
            for f in fields(node):
                v = getattr(node, f.name)
                if isinstance(v, PyIRNode):
                    setattr(
                        node,
                        f.name,
                        RewriteAnonymousFunctionsPass._rw_node(
                            v, ctx=ctx, funcs=funcs, store=False, file=file
                        ),
                    )
                elif isinstance(v, list):
                    setattr(
                        node,
                        f.name,
                        [
                            RewriteAnonymousFunctionsPass._rw_node(
                                x, ctx=ctx, funcs=funcs, store=False, file=file
                            )
                            if isinstance(x, PyIRNode)
                            else x
                            for x in v
                        ],
                    )
                elif isinstance(v, dict):
                    setattr(
                        node,
                        f.name,
                        {
                            k: (
                                RewriteAnonymousFunctionsPass._rw_node(
                                    x, ctx=ctx, funcs=funcs, store=False, file=file
                                )
                                if isinstance(x, PyIRNode)
                                else x
                            )
                            for k, x in v.items()
                        },
                    )
            return node

        return node

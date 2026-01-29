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
        from dataclasses import fields, is_dataclass

        blocked = getattr(ctx, "_anon_blocked_syms", None)
        if blocked is None:
            blocked = set()
            setattr(ctx, "_anon_blocked_syms", blocked)

        def clone_preserve_links(n: PyIRNode) -> PyIRNode:
            pairs: list[tuple[PyIRNode, PyIRNode]] = []

            def clone_value(v):
                if isinstance(v, PyIRNode):
                    return clone_node(v)
                if isinstance(v, list):
                    return [clone_value(x) for x in v]
                if isinstance(v, dict):
                    return {k: clone_value(x) for k, x in v.items()}
                if isinstance(v, tuple):
                    return tuple(clone_value(x) for x in v)
                return v

            def clone_node(x: PyIRNode) -> PyIRNode:
                if not is_dataclass(x):
                    return x
                cls = x.__class__
                kwargs = {f.name: clone_value(getattr(x, f.name)) for f in fields(x)}
                out = cls(**kwargs)  # type: ignore
                pairs.append((x, out))
                return out

            out = clone_node(n)

            for old, new in pairs:
                old_uid = ctx.uid(old)
                new_uid = ctx.uid(new)

                link = ctx.var_links.get(old_uid)
                if link is not None and new_uid not in ctx.var_links:
                    ctx.var_links[new_uid] = link

                node_scope = getattr(ctx, "node_scope", None)
                if isinstance(node_scope, dict):
                    sc = node_scope.get(old_uid)
                    if sc is not None and new_uid not in node_scope:
                        node_scope[new_uid] = sc

            return out

        def make_fn_expr(
            info: _AnonFnInfo, *, line: int | None, offset: int | None
        ) -> PyIRFnExpr:
            sig: dict[str, object] = dict(info.fn.signature)
            for k, v in list(sig.items()):
                if isinstance(v, PyIRNode):
                    sig[k] = clone_preserve_links(v)
                elif isinstance(v, list):
                    sig[k] = [
                        clone_preserve_links(x) if isinstance(x, PyIRNode) else x
                        for x in v
                    ]
                elif isinstance(v, dict):
                    sig[k] = {
                        kk: (clone_preserve_links(x) if isinstance(x, PyIRNode) else x)
                        for kk, x in v.items()
                    }
                elif isinstance(v, tuple):
                    sig[k] = tuple(
                        clone_preserve_links(x) if isinstance(x, PyIRNode) else x
                        for x in v
                    )

            body = [clone_preserve_links(st) for st in info.fn.body]
            expr = PyIRFnExpr(line=line, offset=offset, signature=sig, body=body)

            blocked.add(info.sym_id)
            try:
                expr.body[:] = RewriteAnonymousFunctionsPass._rewrite_stmt_list(
                    expr.body, ctx=ctx, funcs=funcs, file=file
                )
            finally:
                blocked.discard(info.sym_id)

            return expr

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

        if isinstance(node, PyIRSymLink):
            if store or node.is_store:
                return node

            sym_id = int(node.symbol_id)
            if sym_id in blocked:
                return node

            info = funcs.get(sym_id)
            if not info:
                return node

            return make_fn_expr(info, line=node.line, offset=node.offset)

        if isinstance(node, PyIRVarUse):
            if store:
                return node

            link = ctx.var_links.get(ctx.uid(node))
            if not link:
                return node

            sym_id = int(link.target.value)
            if sym_id in blocked:
                return node

            info = funcs.get(sym_id)
            if not info:
                return node

            return make_fn_expr(info, line=node.line, offset=node.offset)

        rewritten_lists: set[int] = set()
        if isinstance(node, PyIRFnExpr):
            node.body[:] = RewriteAnonymousFunctionsPass._rewrite_stmt_list(
                node.body, ctx=ctx, funcs=funcs, file=file
            )
            rewritten_lists.add(id(node.body))

        for lst in MacroUtils.statement_lists(node):
            lst[:] = RewriteAnonymousFunctionsPass._rewrite_stmt_list(
                lst, ctx=ctx, funcs=funcs, file=file
            )
            rewritten_lists.add(id(lst))

        if not is_dataclass(node):
            return node

        cls_name = node.__class__.__name__
        store_fields: set[str] = set()

        if cls_name == "PyIRAssign":
            store_fields = {"targets"}

        elif cls_name in ("PyIRAugAssign", "PyIRAnnAssign"):
            store_fields = {"target"}

        elif cls_name == "PyIRFor":
            store_fields = {"target"}

        elif cls_name == "PyIRWith":
            store_fields = {"targets", "optional_vars", "target"}

        for f in fields(node):
            name = f.name
            if name in ("line", "offset"):
                continue

            value = getattr(node, name)

            if isinstance(value, list) and id(value) in rewritten_lists:
                continue

            field_is_store = name in store_fields

            if isinstance(value, PyIRNode):
                new_child = RewriteAnonymousFunctionsPass._rw_node(
                    value, ctx=ctx, funcs=funcs, store=field_is_store, file=file
                )
                if new_child is not value:
                    setattr(node, name, new_child)

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, PyIRNode):
                        value[i] = RewriteAnonymousFunctionsPass._rw_node(
                            item, ctx=ctx, funcs=funcs, store=field_is_store, file=file
                        )
                    elif isinstance(item, tuple):
                        value[i] = tuple(
                            RewriteAnonymousFunctionsPass._rw_node(
                                x, ctx=ctx, funcs=funcs, store=False, file=file
                            )
                            if isinstance(x, PyIRNode)
                            else x
                            for x in item
                        )

            elif isinstance(value, dict):
                for k, v in list(value.items()):
                    if isinstance(v, PyIRNode):
                        value[k] = RewriteAnonymousFunctionsPass._rw_node(
                            v, ctx=ctx, funcs=funcs, store=False, file=file
                        )

            elif isinstance(value, tuple):
                new_t = tuple(
                    RewriteAnonymousFunctionsPass._rw_node(
                        x, ctx=ctx, funcs=funcs, store=False, file=file
                    )
                    if isinstance(x, PyIRNode)
                    else x
                    for x in value
                )
                if new_t is not value:
                    setattr(node, name, new_t)

        return node

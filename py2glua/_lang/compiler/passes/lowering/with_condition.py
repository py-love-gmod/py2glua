from __future__ import annotations

from typing import Final

from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRDecorator,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ...compiler_ir import PyIRFunctionExpr
from ..rewrite_utils import rewrite_dataclass_children, rewrite_stmt_block
from ..analysis.symlinks import PyIRSymLink, SymLinkContext
from ..common import (
    canon_path,
    collect_decl_symid_cache,
    collect_core_compiler_directive_local_symbol_ids,
    is_attr_on_symbol_ids,
    resolve_symbol_ids_by_attr_chain,
)


class CollectWithConditionClassesPass:
    """
    Lowering-pass (phase 1):
    Collects symids of classes decorated with @CompilerDirective.with_condition(),
    but ONLY when that decorator is proven to come from core CompilerDirective.

    Writes into ctx:
      - ctx._with_condition_class_symids: set[int]
      - ctx._with_condition_symid_by_decl_cache: dict[(Path, line, name) -> symid]
    """

    _DECL_CACHE_FIELD: Final[str] = "_with_condition_symid_by_decl_cache"
    _WITH_COND_NAME: Final[str] = "with_condition"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError(
                "CollectWithConditionClassesPass.run РѕР¶РёРґР°РµС‚ PyIRFile"
            )

        s = ctx._with_condition_class_symids

        if ir.path is None:
            return ir

        # Needed to resolve module name -> local exported symids.
        ctx.ensure_module_index()
        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        symid_by_decl = collect_decl_symid_cache(ctx, cache_field=cls._DECL_CACHE_FIELD)
        fp = canon_path(ir.path)
        if fp is None:
            return ir

        # Collect only core CompilerDirective bindings visible in this file.
        cd_local_symids = collect_core_compiler_directive_local_symbol_ids(
            ir=ir,
            ctx=ctx,
            current_module=current_module,
        )
        # Scan classes; collect those decorated with *CompilerDirective*.with_condition()
        for n in ir.walk():
            if not isinstance(n, PyIRClassDef):
                continue
            if n.line is None:
                continue

            if not any(
                is_attr_on_symbol_ids(
                    d.exper,
                    attr=cls._WITH_COND_NAME,
                    symbol_ids=cd_local_symids,
                    ctx=ctx,
                )
                for d in (n.decorators or [])
            ):
                continue

            class_symid = symid_by_decl.get((fp, n.line, n.name))
            if class_symid is None:
                continue

            s.add(int(class_symid))

            # Strip only the matched directive decorator (not any random ".with_condition")
            n.decorators = [
                d
                for d in n.decorators
                if not is_attr_on_symbol_ids(
                    d.exper,
                    attr=cls._WITH_COND_NAME,
                    symbol_ids=cd_local_symids,
                    ctx=ctx,
                )
            ]

        return ir


class RewriteWithConditionBlocksPass:
    """
    Lowering-pass (phase 2):
    Rewrites:

        with Class.attr:
            body

    into:

        if Class.attr:
            body

    Condition source is any class symid collected by CollectWithConditionClassesPass.
    Supports multiple with-items by nesting (right-to-left):
        with A.x, B.y:
            body
      => if A.x:
            if B.y:
                body

    If with has mixed items (condition + other), it nests as:
        if Cond:
            with Other:
                body
    """

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError(
                "RewriteWithConditionBlocksPass.run РѕР¶РёРґР°РµС‚ PyIRFile"
            )

        s = ctx._with_condition_class_symids
        if not s:
            return ir

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            if isinstance(st, PyIRFunctionDef):
                st.decorators = [
                    d
                    for d in (rw_expr(d) for d in st.decorators)
                    if isinstance(d, PyIRDecorator)
                ]
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRFunctionExpr):
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRClassDef):
                st.decorators = [
                    d
                    for d in (rw_expr(d) for d in st.decorators)
                    if isinstance(d, PyIRDecorator)
                ]
                st.bases = [rw_expr(b) for b in st.bases]
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRIf):
                st.test = rw_expr(st.test)
                st.body = rw_block(st.body)
                st.orelse = rw_block(st.orelse)
                return [st]

            if isinstance(st, PyIRWhile):
                st.test = rw_expr(st.test)
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRFor):
                st.target = rw_expr(st.target)
                st.iter = rw_expr(st.iter)
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRWith):
                inner = rw_block(st.body)

                rebuilt: list[PyIRNode] = inner
                for it in reversed(st.items):
                    rebuilt = cls._wrap_with_item(
                        it,
                        rebuilt,
                        st,
                        ctx=ctx,
                        with_condition_class_symids=s,
                    )

                return rebuilt

            return [rw_generic_stmt(st)]

        def rw_generic_stmt(node: PyIRNode) -> PyIRNode:
            if isinstance(node, PyIRAssign):
                node.targets = [rw_expr(t) for t in node.targets]
                node.value = rw_expr(node.value)
                return node

            if isinstance(node, PyIRAugAssign):
                node.target = rw_expr(node.target)
                node.value = rw_expr(node.value)
                return node

            rewrite_dataclass_children(node, rw_expr)

            return node

        def rw_expr(node: PyIRNode) -> PyIRNode:
            if isinstance(node, PyIRAttribute):
                node.value = rw_expr(node.value)
                return node

            if isinstance(node, PyIRCall):
                node.func = rw_expr(node.func)
                node.args_p = [rw_expr(a) for a in node.args_p]
                node.args_kw = {k: rw_expr(v) for k, v in node.args_kw.items()}
                return node

            rewrite_dataclass_children(node, rw_expr)
            return node

        ir.body = rw_block(ir.body)
        return ir

    @staticmethod
    def _is_with_condition_item(
        it: PyIRWithItem,
        *,
        ctx: SymLinkContext,
        with_condition_class_symids: set[int],
    ) -> bool:
        ce = it.context_expr
        if not isinstance(ce, PyIRAttribute):
            return False

        if isinstance(ce.value, PyIRSymLink):
            return int(ce.value.symbol_id) in with_condition_class_symids

        resolved = resolve_symbol_ids_by_attr_chain(ctx, ce.value)
        return any(int(sid) in with_condition_class_symids for sid in resolved)

    @classmethod
    def _wrap_with_item(
        cls,
        it: PyIRWithItem,
        inner: list[PyIRNode],
        src_with: PyIRWith,
        *,
        ctx: SymLinkContext,
        with_condition_class_symids: set[int],
    ) -> list[PyIRNode]:
        if it.optional_vars is not None:
            w = PyIRWith(
                line=src_with.line,
                offset=src_with.offset,
                items=[it],
                body=inner,
            )
            return [w]

        if cls._is_with_condition_item(
            it,
            ctx=ctx,
            with_condition_class_symids=with_condition_class_symids,
        ):
            test = it.context_expr
            out_if = PyIRIf(
                line=src_with.line,
                offset=src_with.offset,
                test=test,
                body=inner,
                orelse=[],
            )
            return [out_if]

        w = PyIRWith(
            line=src_with.line,
            offset=src_with.offset,
            items=[it],
            body=inner,
        )
        return [w]

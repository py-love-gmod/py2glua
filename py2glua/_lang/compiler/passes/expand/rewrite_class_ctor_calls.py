from __future__ import annotations

from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRFile,
    PyIRNode,
    PyIRVarUse,
)
from ..rewrite_utils import rewrite_expr_with_store_default, rewrite_stmt_block
from .expand_context import ExpandContext


class RewriteClassCtorCallsPass:
    """
    Переписывает `ClassName(...)` в `ClassName.__init__(...)`.

    Ограничение:
      - переписываются только вызовы классов, объявленных в текущем файле
        (top-level var name, который раньше был `PyIRClassDef`).
    """

    _INIT_ATTR = "__init__"

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        _ = ctx

        local_classes = {st.name for st in ir.body if isinstance(st, PyIRClassDef)}

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(expr: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(expr, PyIRCall):
                expr.func = rw_expr(expr.func, False)
                expr.args_p = [rw_expr(a, False) for a in expr.args_p]
                expr.args_kw = {k: rw_expr(v, False) for k, v in expr.args_kw.items()}

                if not store and isinstance(expr.func, PyIRVarUse):
                    if expr.func.name in local_classes:
                        base = expr.func
                        expr.func = PyIRAttribute(
                            line=base.line,
                            offset=base.offset,
                            value=base,
                            attr=RewriteClassCtorCallsPass._INIT_ATTR,
                        )

                return expr

            return rewrite_expr_with_store_default(
                expr,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

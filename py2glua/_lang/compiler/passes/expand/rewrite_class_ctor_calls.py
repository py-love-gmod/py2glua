from __future__ import annotations

from typing import List

from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBinOP,
    PyIRBreak,
    PyIRCall,
    PyIRClassDef,
    PyIRComment,
    PyIRConstant,
    PyIRContinue,
    PyIRDecorator,
    PyIRDict,
    PyIRDictItem,
    PyIREmitExpr,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRList,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
    PyIRSet,
    PyIRSubscript,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarCreate,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ...compiler_ir import PyIRFunctionExpr
from .expand_context import ExpandContext


class RewriteClassCtorCallsPass:
    """
    Переписывает "ClassName(...)" в "ClassName.__init__(...)".

    Это УМЫШЛЕННО не аллокация объекта и не sugar new().
    Цель — убрать магию вызова класса как функции, чтобы дальше
    всё было одинаково (вызов обычной функции/атрибута).

    Ограничение:
      - переписывает только вызовы классов, объявленных в текущем файле
        (top-level PyIRClassDef).
    """

    _INIT_ATTR = "__init__"

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        _ = ctx

        # collect top-level class names in this file
        local_classes = {st.name for st in ir.body if isinstance(st, PyIRClassDef)}

        ir.body = RewriteClassCtorCallsPass._rw_stmt_list(local_classes, ir.body)
        return ir

    @staticmethod
    def _rw_stmt_list(local_classes: set[str], stmts: List[PyIRNode]) -> List[PyIRNode]:
        out: List[PyIRNode] = []
        for st in stmts:
            out.extend(RewriteClassCtorCallsPass._rw_stmt(local_classes, st))
        return out

    @staticmethod
    def _rw_stmt(local_classes: set[str], st: PyIRNode) -> List[PyIRNode]:
        if isinstance(st, PyIRIf):
            st.test = RewriteClassCtorCallsPass._rw_expr(local_classes, st.test)
            st.body = RewriteClassCtorCallsPass._rw_stmt_list(local_classes, st.body)
            st.orelse = RewriteClassCtorCallsPass._rw_stmt_list(
                local_classes, st.orelse
            )
            return [st]

        if isinstance(st, PyIRWhile):
            st.test = RewriteClassCtorCallsPass._rw_expr(local_classes, st.test)
            st.body = RewriteClassCtorCallsPass._rw_stmt_list(local_classes, st.body)
            return [st]

        if isinstance(st, PyIRFor):
            st.target = RewriteClassCtorCallsPass._rw_expr(local_classes, st.target)
            st.iter = RewriteClassCtorCallsPass._rw_expr(local_classes, st.iter)
            st.body = RewriteClassCtorCallsPass._rw_stmt_list(local_classes, st.body)
            return [st]

        if isinstance(st, PyIRWithItem):
            st.context_expr = RewriteClassCtorCallsPass._rw_expr(
                local_classes, st.context_expr
            )
            if st.optional_vars is not None:
                st.optional_vars = RewriteClassCtorCallsPass._rw_expr(
                    local_classes, st.optional_vars
                )
            return [st]

        if isinstance(st, PyIRWith):
            st.items = [  # type: ignore[list-item]
                RewriteClassCtorCallsPass._rw_expr(local_classes, it) for it in st.items
            ]
            st.body = RewriteClassCtorCallsPass._rw_stmt_list(local_classes, st.body)
            return [st]

        if isinstance(st, PyIRAssign):
            st.targets = [
                RewriteClassCtorCallsPass._rw_expr(local_classes, t) for t in st.targets
            ]
            st.value = RewriteClassCtorCallsPass._rw_expr(local_classes, st.value)
            return [st]

        if isinstance(st, PyIRAugAssign):
            st.target = RewriteClassCtorCallsPass._rw_expr(local_classes, st.target)
            st.value = RewriteClassCtorCallsPass._rw_expr(local_classes, st.value)
            return [st]

        if isinstance(st, PyIRReturn):
            st.value = RewriteClassCtorCallsPass._rw_expr(local_classes, st.value)
            return [st]

        if isinstance(st, PyIRFunctionDef):
            st.decorators = [  # type: ignore[list-item]
                RewriteClassCtorCallsPass._rw_expr(local_classes, d)
                for d in st.decorators
            ]
            st.body = RewriteClassCtorCallsPass._rw_stmt_list(local_classes, st.body)
            return [st]

        if isinstance(st, PyIRFunctionExpr):
            st.body = RewriteClassCtorCallsPass._rw_stmt_list(local_classes, st.body)
            return [st]

        if isinstance(st, PyIRClassDef):
            st.bases = [
                RewriteClassCtorCallsPass._rw_expr(local_classes, b) for b in st.bases
            ]
            st.decorators = [  # type: ignore[list-item]
                RewriteClassCtorCallsPass._rw_expr(local_classes, d)
                for d in st.decorators
            ]
            st.body = RewriteClassCtorCallsPass._rw_stmt_list(local_classes, st.body)
            return [st]

        if isinstance(
            st,
            (
                PyIRImport,
                PyIRBreak,
                PyIRContinue,
                PyIRPass,
                PyIRComment,
                PyIRVarCreate,
                PyIRConstant,
                PyIRVarUse,
            ),
        ):
            return [st]

        # expression-statement fallback
        return [RewriteClassCtorCallsPass._rw_expr(local_classes, st)]

    @staticmethod
    def _rw_expr(local_classes: set[str], expr: PyIRNode) -> PyIRNode:
        if isinstance(
            expr,
            (
                PyIRConstant,
                PyIRVarUse,
                PyIRVarCreate,
                PyIRImport,
                PyIRBreak,
                PyIRContinue,
                PyIRPass,
                PyIRComment,
            ),
        ):
            return expr

        if isinstance(expr, PyIRUnaryOP):
            expr.value = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.value)
            return expr

        if isinstance(expr, PyIRBinOP):
            expr.left = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.left)
            expr.right = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.right)
            return expr

        if isinstance(expr, PyIRAttribute):
            expr.value = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.value)
            return expr

        if isinstance(expr, PyIRSubscript):
            expr.value = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.value)
            expr.index = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.index)
            return expr

        if isinstance(expr, PyIRList):
            expr.elements = [
                RewriteClassCtorCallsPass._rw_expr(local_classes, e)
                for e in expr.elements
            ]
            return expr

        if isinstance(expr, PyIRTuple):
            expr.elements = [
                RewriteClassCtorCallsPass._rw_expr(local_classes, e)
                for e in expr.elements
            ]
            return expr

        if isinstance(expr, PyIRSet):
            expr.elements = [
                RewriteClassCtorCallsPass._rw_expr(local_classes, e)
                for e in expr.elements
            ]
            return expr

        if isinstance(expr, PyIRDictItem):
            expr.key = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.key)
            expr.value = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.value)
            return expr

        if isinstance(expr, PyIRDict):
            expr.items = [  # type: ignore[list-item]
                RewriteClassCtorCallsPass._rw_expr(local_classes, i) for i in expr.items
            ]
            return expr

        if isinstance(expr, PyIRDecorator):
            expr.exper = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.exper)
            expr.args_p = [
                RewriteClassCtorCallsPass._rw_expr(local_classes, a)
                for a in expr.args_p
            ]
            expr.args_kw = {
                k: RewriteClassCtorCallsPass._rw_expr(local_classes, v)
                for k, v in expr.args_kw.items()
            }
            return expr

        if isinstance(expr, PyIREmitExpr):
            expr.args_p = [
                RewriteClassCtorCallsPass._rw_expr(local_classes, a)
                for a in expr.args_p
            ]
            expr.args_kw = {
                k: RewriteClassCtorCallsPass._rw_expr(local_classes, v)
                for k, v in expr.args_kw.items()
            }
            return expr

        if isinstance(expr, PyIRCall):
            expr.func = RewriteClassCtorCallsPass._rw_expr(local_classes, expr.func)
            expr.args_p = [
                RewriteClassCtorCallsPass._rw_expr(local_classes, a)
                for a in expr.args_p
            ]
            expr.args_kw = {
                k: RewriteClassCtorCallsPass._rw_expr(local_classes, v)
                for k, v in expr.args_kw.items()
            }

            # rewrite: ClassName(...) -> ClassName.__init__(...)
            if isinstance(expr.func, PyIRVarUse) and expr.func.name in local_classes:
                base = expr.func
                expr.func = PyIRAttribute(
                    line=base.line,
                    offset=base.offset,
                    value=base,
                    attr=RewriteClassCtorCallsPass._INIT_ATTR,
                )

            return expr

        return expr

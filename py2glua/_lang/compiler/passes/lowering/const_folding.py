from __future__ import annotations

from math import floor
from typing import Final, cast

from ....compiler.passes.analysis.symlinks import SymLinkContext
from ....py.ir_dataclass import (
    LuaNil,
    PyBinOPType,
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
    PyUnaryOPType,
)


class ConstFoldingPass:
    """
    Expression-only constant folding under Lua-like semantics.

    - Folds PyIRUnaryOP / PyIRBinOP when operands are constants
    - Rewrites expressions inside statements (including PyIRIf.test / PyIRWhile.test)
    - DOES NOT remove/replace statements (no branch elimination, no loop deletion)

    Extra optimization:
      - short-circuit simplification for AND/OR when LEFT side is constant:
          (truthy_const and X)  -> X
          (falsy_const  and X)  -> falsy_const
          (truthy_const or  X)  -> truthy_const
          (falsy_const  or  X)  -> X
        This is still expression folding (no statement elimination) and is safe
        because left side is a constant (no side effects).
    """

    _NO: Final[object] = object()

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        _ = ctx

        def is_const_value(v: object | LuaNil) -> bool:
            return v is LuaNil or v is None or isinstance(v, (bool, int, float, str))

        def lua_truthy(v: object | LuaNil) -> bool:
            return not (v is False or v is LuaNil or v is None)

        def is_num(v: object | LuaNil) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        def is_int(v: object | LuaNil) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        def make_const(ref: PyIRNode, value: object | LuaNil) -> PyIRConstant:
            return PyIRConstant(
                line=ref.line,
                offset=ref.offset,
                value=value,
            )

        def const_value(node: PyIRNode) -> object | LuaNil | object:
            if isinstance(node, PyIRConstant) and is_const_value(node.value):
                return node.value

            return ConstFoldingPass._NO

        def eq_lua(a: object | LuaNil, b: object | LuaNil) -> bool:
            if a is LuaNil:
                return b is LuaNil

            if a is None:
                return b is None

            return a == b

        def try_eval_unary(
            op: PyUnaryOPType, v: object | LuaNil
        ) -> object | LuaNil | object:
            if op == PyUnaryOPType.NOT:
                return not lua_truthy(v)

            if op == PyUnaryOPType.PLUS:
                if is_num(v):
                    return v

                return ConstFoldingPass._NO

            if op == PyUnaryOPType.MINUS:
                if is_num(v):
                    return -cast(float | int, v)

                return ConstFoldingPass._NO

            if op == PyUnaryOPType.BIT_INV:
                if is_int(v):
                    return ~cast(int, v)

                return ConstFoldingPass._NO

            return ConstFoldingPass._NO

        def try_eval_bin(
            op: PyBinOPType, a: object | LuaNil, b: object | LuaNil
        ) -> object | LuaNil | object:
            if op == PyBinOPType.AND:
                return b if lua_truthy(a) else a

            if op == PyBinOPType.OR:
                return a if lua_truthy(a) else b

            if op == PyBinOPType.EQ:
                return eq_lua(a, b)

            if op == PyBinOPType.NE:
                return not eq_lua(a, b)

            if op in (PyBinOPType.LT, PyBinOPType.LE, PyBinOPType.GT, PyBinOPType.GE):
                if is_num(a) and is_num(b):
                    fa = float(cast(float | int, a))
                    fb = float(cast(float | int, b))

                    if op == PyBinOPType.LT:
                        return fa < fb

                    if op == PyBinOPType.LE:
                        return fa <= fb

                    if op == PyBinOPType.GT:
                        return fa > fb

                    if op == PyBinOPType.GE:
                        return fa >= fb

                    return ConstFoldingPass._NO

                if isinstance(a, str) and isinstance(b, str):
                    if op == PyBinOPType.LT:
                        return a < b

                    if op == PyBinOPType.LE:
                        return a <= b

                    if op == PyBinOPType.GT:
                        return a > b

                    if op == PyBinOPType.GE:
                        return a >= b

                    return ConstFoldingPass._NO

                return ConstFoldingPass._NO

            if op in (
                PyBinOPType.ADD,
                PyBinOPType.SUB,
                PyBinOPType.MUL,
                PyBinOPType.DIV,
                PyBinOPType.FLOORDIV,
                PyBinOPType.MOD,
                PyBinOPType.POW,
            ):
                if not (is_num(a) and is_num(b)):
                    return ConstFoldingPass._NO

                aa = float(cast(float | int, a))
                bb = float(cast(float | int, b))

                if (
                    op in (PyBinOPType.DIV, PyBinOPType.FLOORDIV, PyBinOPType.MOD)
                    and bb == 0.0
                ):
                    return ConstFoldingPass._NO

                if op == PyBinOPType.ADD:
                    if is_int(a) and is_int(b):
                        return cast(int, a) + cast(int, b)

                    return aa + bb

                if op == PyBinOPType.SUB:
                    if is_int(a) and is_int(b):
                        return cast(int, a) - cast(int, b)

                    return aa - bb

                if op == PyBinOPType.MUL:
                    if is_int(a) and is_int(b):
                        return cast(int, a) * cast(int, b)

                    return aa * bb

                if op == PyBinOPType.DIV:
                    return aa / bb

                if op == PyBinOPType.FLOORDIV:
                    return int(floor(aa / bb))

                if op == PyBinOPType.MOD:
                    if is_int(a) and is_int(b):
                        return cast(int, a) % cast(int, b)

                    return aa % bb

                if op == PyBinOPType.POW:
                    if is_int(a) and is_int(b) and cast(int, b) >= 0:
                        return cast(int, a) ** cast(int, b)

                    return aa**bb

            if op in (
                PyBinOPType.BIT_OR,
                PyBinOPType.BIT_XOR,
                PyBinOPType.BIT_AND,
                PyBinOPType.BIT_LSHIFT,
                PyBinOPType.BIT_RSHIFT,
            ):
                if not (is_int(a) and is_int(b)):
                    return ConstFoldingPass._NO

                ia = cast(int, a)
                ib = cast(int, b)
                if ib < 0:
                    return ConstFoldingPass._NO

                if op == PyBinOPType.BIT_OR:
                    return ia | ib
                if op == PyBinOPType.BIT_XOR:
                    return ia ^ ib
                if op == PyBinOPType.BIT_AND:
                    return ia & ib
                if op == PyBinOPType.BIT_LSHIFT:
                    return ia << ib
                if op == PyBinOPType.BIT_RSHIFT:
                    return ia >> ib

            return ConstFoldingPass._NO

        def fold_expr(node: PyIRNode) -> PyIRNode:
            if isinstance(
                node,
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
                return node

            if isinstance(node, PyIRUnaryOP):
                node.value = fold_expr(node.value)
                v = const_value(node.value)
                if v is not ConstFoldingPass._NO:
                    out = try_eval_unary(node.op, cast(object | LuaNil, v))
                    if out is not ConstFoldingPass._NO:
                        return make_const(node, cast(object | LuaNil, out))
                return node

            if isinstance(node, PyIRBinOP):
                node.left = fold_expr(node.left)
                node.right = fold_expr(node.right)

                if node.op in (PyBinOPType.AND, PyBinOPType.OR):
                    lv = const_value(node.left)
                    if lv is not ConstFoldingPass._NO:
                        lcv = cast(object | LuaNil, lv)

                        if node.op == PyBinOPType.AND:
                            if not lua_truthy(lcv):
                                return make_const(node, lcv)

                            return node.right

                        if lua_truthy(lcv):
                            return make_const(node, lcv)

                        return node.right

                a = const_value(node.left)
                b = const_value(node.right)
                if a is not ConstFoldingPass._NO and b is not ConstFoldingPass._NO:
                    out = try_eval_bin(
                        node.op,
                        cast(object | LuaNil, a),
                        cast(object | LuaNil, b),
                    )
                    if out is not ConstFoldingPass._NO:
                        return make_const(node, cast(object | LuaNil, out))

                return node

            if isinstance(node, PyIRAttribute):
                node.value = fold_expr(node.value)
                return node

            if isinstance(node, PyIRSubscript):
                node.value = fold_expr(node.value)
                node.index = fold_expr(node.index)
                return node

            if isinstance(node, PyIRList):
                node.elements = [fold_expr(e) for e in node.elements]
                return node

            if isinstance(node, PyIRTuple):
                node.elements = [fold_expr(e) for e in node.elements]
                return node

            if isinstance(node, PyIRSet):
                node.elements = [fold_expr(e) for e in node.elements]
                return node

            if isinstance(node, PyIRDictItem):
                node.key = fold_expr(node.key)
                node.value = fold_expr(node.value)
                return node

            if isinstance(node, PyIRDict):
                node.items = [cast(PyIRDictItem, fold_expr(i)) for i in node.items]
                return node

            if isinstance(node, PyIRCall):
                node.func = fold_expr(node.func)
                node.args_p = [fold_expr(a) for a in node.args_p]
                node.args_kw = {k: fold_expr(v) for k, v in node.args_kw.items()}
                return node

            if isinstance(node, PyIRDecorator):
                node.exper = fold_expr(node.exper)
                node.args_p = [fold_expr(a) for a in node.args_p]
                node.args_kw = {k: fold_expr(v) for k, v in node.args_kw.items()}
                return node

            if PyIREmitExpr is not None and isinstance(node, PyIREmitExpr):
                node.args_p = [fold_expr(a) for a in node.args_p]
                node.args_kw = {k: fold_expr(v) for k, v in node.args_kw.items()}
                return node

            return node

        def fold_stmt_list(stmts: list[PyIRNode]) -> list[PyIRNode]:
            out: list[PyIRNode] = []
            for s in stmts:
                out.extend(fold_stmt(s))

            return out

        def fold_stmt(node: PyIRNode) -> list[PyIRNode]:
            if isinstance(node, PyIRIf):
                node.test = fold_expr(node.test)
                node.body = fold_stmt_list(node.body)
                node.orelse = fold_stmt_list(node.orelse)
                return [node]

            if isinstance(node, PyIRWhile):
                node.test = fold_expr(node.test)
                node.body = fold_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRFor):
                node.target = fold_expr(node.target)
                node.iter = fold_expr(node.iter)
                node.body = fold_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRWithItem):
                node.context_expr = fold_expr(node.context_expr)
                if node.optional_vars is not None:
                    node.optional_vars = fold_expr(node.optional_vars)
                return [node]

            if isinstance(node, PyIRWith):
                node.items = [cast(PyIRWithItem, fold_expr(i)) for i in node.items]
                node.body = fold_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRAssign):
                node.targets = [fold_expr(t) for t in node.targets]
                node.value = fold_expr(node.value)
                return [node]

            if isinstance(node, PyIRAugAssign):
                node.target = fold_expr(node.target)
                node.value = fold_expr(node.value)
                return [node]

            if isinstance(node, PyIRReturn):
                if node.value is not None:
                    node.value = fold_expr(node.value)
                return [node]

            if isinstance(node, PyIRFunctionDef):
                node.decorators = [
                    cast(PyIRDecorator, fold_expr(d)) for d in node.decorators
                ]
                node.body = fold_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRClassDef):
                node.bases = [fold_expr(b) for b in node.bases]
                node.decorators = [
                    cast(PyIRDecorator, fold_expr(d)) for d in node.decorators
                ]
                node.body = fold_stmt_list(node.body)
                return [node]

            if isinstance(
                node,
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
                return [node]

            return [fold_expr(node)]

        ir.body = fold_stmt_list(ir.body)
        return ir

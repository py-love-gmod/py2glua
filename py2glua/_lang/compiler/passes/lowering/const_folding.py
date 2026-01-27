from __future__ import annotations

from dataclasses import fields, is_dataclass

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyBinOPType,
    PyIRBinOP,
    PyIRConstant,
    PyIRNode,
    PyIRUnaryOP,
    PyUnaryOPType,
)
from ..analysis.ctx import AnalysisContext, AnalysisPass


class ConstFoldingPass(AnalysisPass):
    """
    Базовый const folding для простых арифметических/логических выражений.

    Правила:
    - Фолдим только если все операнды являются PyIRConstant.
    - Никаких вычислений для переменных/вызовов/атрибутов.
    - Логика AND/OR фолдится только если оба операнда bool (во избежание различий Python/Lua).
    - Сравнения фолдятся для int/float/str/bool (если Python умеет сравнить без исключения).
    """

    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        ir.body = cls._rewrite_block(ir.body)

    @classmethod
    def _rewrite_block(cls, body: list[PyIRNode]) -> list[PyIRNode]:
        out: list[PyIRNode] = []
        for n in body:
            out.append(cls._rewrite_node(n))

        return out

    @classmethod
    def _rewrite_node(cls, node: PyIRNode) -> PyIRNode:
        if isinstance(node, PyIRBinOP):
            return cls._fold_binop(node)

        if isinstance(node, PyIRUnaryOP):
            return cls._fold_unary(node)

        if not is_dataclass(node):
            return node

        for f in fields(node):
            val = getattr(node, f.name)
            new_val = cls._rewrite_value(val)
            if new_val is not val:
                setattr(node, f.name, new_val)

        return node

    @classmethod
    def _rewrite_value(cls, val):
        if isinstance(val, PyIRNode):
            return cls._rewrite_node(val)

        if isinstance(val, list):
            return [cls._rewrite_value(v) for v in val]

        if isinstance(val, tuple):
            return tuple(cls._rewrite_value(v) for v in val)

        if isinstance(val, dict):
            return {
                cls._rewrite_value(k): cls._rewrite_value(v) for k, v in val.items()
            }

        return val

    @classmethod
    def _fold_unary(cls, node: PyIRUnaryOP) -> PyIRNode:
        v = cls._rewrite_node(node.value)
        node.value = v

        if not isinstance(v, PyIRConstant):
            return node

        x = v.value

        try:
            if node.op == PyUnaryOPType.PLUS:
                if isinstance(x, (int, float, bool)):
                    return PyIRConstant(node.line, node.offset, +x)

            if node.op == PyUnaryOPType.MINUS:
                if isinstance(x, (int, float, bool)):
                    return PyIRConstant(node.line, node.offset, -x)

            if node.op == PyUnaryOPType.NOT:
                if isinstance(x, bool):
                    return PyIRConstant(node.line, node.offset, (not x))

            if node.op == PyUnaryOPType.BIT_INV:
                if isinstance(x, (int, bool)):
                    return PyIRConstant(node.line, node.offset, ~int(x))

        except Exception:
            return node

        return node

    @classmethod
    def _fold_binop(cls, node: PyIRBinOP) -> PyIRNode:
        left = cls._rewrite_node(node.left)
        right = cls._rewrite_node(node.right)
        node.left = left
        node.right = right

        if not isinstance(left, PyIRConstant) or not isinstance(right, PyIRConstant):
            return node

        a = left.value
        b = right.value

        try:
            # арифметика
            if node.op == PyBinOPType.ADD:
                if isinstance(a, (int, float, bool, str)) and isinstance(
                    b, (int, float, bool, str)
                ):
                    return PyIRConstant(node.line, node.offset, a + b)  # type: ignore

            if node.op == PyBinOPType.SUB:
                if isinstance(a, (int, float, bool)) and isinstance(
                    b, (int, float, bool)
                ):
                    return PyIRConstant(node.line, node.offset, a - b)

            if node.op == PyBinOPType.MUL:
                if isinstance(a, (int, float, bool)) and isinstance(
                    b, (int, float, bool)
                ):
                    return PyIRConstant(node.line, node.offset, a * b)

            if node.op == PyBinOPType.DIV:
                if isinstance(a, (int, float, bool)) and isinstance(
                    b, (int, float, bool)
                ):
                    return PyIRConstant(node.line, node.offset, a / b)

            if node.op == PyBinOPType.FLOORDIV:
                if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
                    return PyIRConstant(node.line, node.offset, a // b)

            if node.op == PyBinOPType.MOD:
                if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
                    return PyIRConstant(node.line, node.offset, a % b)

            if node.op == PyBinOPType.POW:
                if isinstance(a, (int, float, bool)) and isinstance(
                    b, (int, float, bool)
                ):
                    return PyIRConstant(node.line, node.offset, a**b)

            # сравнения
            if node.op == PyBinOPType.EQ:
                return PyIRConstant(node.line, node.offset, a == b)

            if node.op == PyBinOPType.NE:
                return PyIRConstant(node.line, node.offset, a != b)

            if node.op == PyBinOPType.LT:
                return PyIRConstant(node.line, node.offset, a < b)  # type: ignore

            if node.op == PyBinOPType.GT:
                return PyIRConstant(node.line, node.offset, a > b)  # type: ignore

            if node.op == PyBinOPType.LE:
                return PyIRConstant(node.line, node.offset, a <= b)  # type: ignore

            if node.op == PyBinOPType.GE:
                return PyIRConstant(node.line, node.offset, a >= b)  # type: ignore

            # логика
            # В Lua truthy/falsy отличается от Python. Поэтому фолдим AND/OR только для bool/bool.
            if node.op == PyBinOPType.AND:
                if isinstance(a, bool) and isinstance(b, bool):
                    return PyIRConstant(node.line, node.offset, (a and b))

            if node.op == PyBinOPType.OR:
                if isinstance(a, bool) and isinstance(b, bool):
                    return PyIRConstant(node.line, node.offset, (a or b))

            # битовые
            if node.op == PyBinOPType.BIT_AND:
                if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
                    return PyIRConstant(node.line, node.offset, int(a) & int(b))

            if node.op == PyBinOPType.BIT_OR:
                if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
                    return PyIRConstant(node.line, node.offset, int(a) | int(b))

            if node.op == PyBinOPType.BIT_XOR:
                if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
                    return PyIRConstant(node.line, node.offset, int(a) ^ int(b))

            if node.op == PyBinOPType.BIT_LSHIFT:
                if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
                    return PyIRConstant(node.line, node.offset, int(a) << int(b))

            if node.op == PyBinOPType.BIT_RSHIFT:
                if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
                    return PyIRConstant(node.line, node.offset, int(a) >> int(b))

        except Exception:
            return node

        return node

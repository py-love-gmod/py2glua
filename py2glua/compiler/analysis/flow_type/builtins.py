from __future__ import annotations

from plg_reader import IRBinOpType

from .types import (
    AnyType,
    BoolType,
    BytesType,
    DictType,
    FloatType,
    IntType,
    ListType,
    SetType,
    StrType,
    Type,
)

_binop_table: dict[tuple[Type, IRBinOpType, Type], Type] = {}


def register_binop(left: Type, op: IRBinOpType, right: Type, result: Type):
    _binop_table[(left, op, right)] = result


def get_binop_result(left: Type, op: IRBinOpType, right: Type) -> Type | None:

    if (left, op, right) in _binop_table:
        return _binop_table[(left, op, right)]

    if op in _commutative_ops and (right, op, left) in _binop_table:
        return _binop_table[(right, op, left)]

    if left is AnyType or right is AnyType:
        return AnyType

    return None


_commutative_ops = {
    IRBinOpType.ADD,
    IRBinOpType.MUL,
    IRBinOpType.EQ,
    IRBinOpType.NE,
    IRBinOpType.AND,
    IRBinOpType.OR,
    IRBinOpType.BIT_AND,
    IRBinOpType.BIT_OR,
    IRBinOpType.BIT_XOR,
}


for int_float_op in (IRBinOpType.ADD, IRBinOpType.SUB, IRBinOpType.MUL):
    register_binop(IntType, int_float_op, IntType, IntType)
    register_binop(IntType, int_float_op, FloatType, FloatType)
    register_binop(FloatType, int_float_op, IntType, FloatType)
    register_binop(FloatType, int_float_op, FloatType, FloatType)

register_binop(IntType, IRBinOpType.DIV, IntType, FloatType)
register_binop(IntType, IRBinOpType.DIV, FloatType, FloatType)
register_binop(FloatType, IRBinOpType.DIV, IntType, FloatType)
register_binop(FloatType, IRBinOpType.DIV, FloatType, FloatType)

register_binop(StrType, IRBinOpType.ADD, StrType, StrType)
register_binop(BytesType, IRBinOpType.ADD, BytesType, BytesType)

for cmp_op in (
    IRBinOpType.EQ,
    IRBinOpType.NE,
    IRBinOpType.LT,
    IRBinOpType.LE,
    IRBinOpType.GT,
    IRBinOpType.GE,
):
    register_binop(IntType, cmp_op, IntType, BoolType)
    register_binop(FloatType, cmp_op, FloatType, BoolType)
    register_binop(StrType, cmp_op, StrType, BoolType)
    register_binop(BytesType, cmp_op, BytesType, BoolType)

register_binop(BoolType, IRBinOpType.AND, BoolType, BoolType)
register_binop(BoolType, IRBinOpType.OR, BoolType, BoolType)


def get_iteration_element_type(iter_type: Type) -> Type:
    if isinstance(iter_type, ListType):
        return iter_type.item

    if isinstance(iter_type, SetType):
        return iter_type.item

    if isinstance(iter_type, DictType):
        return iter_type.key

    return AnyType

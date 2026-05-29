from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from plg_reader import (
    IRCall,
    IRConstant,
    IRFunctionDef,
    IRName,
    IRNode,
    IRSubscript,
    IRTuple,
)

from ..symtable import IRSymbolRef, Scope, SymbolTable
from .builtins import (
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
from .types import (
    CallableType,
    ClassType,
    EllipsisType,
    NoneType,
    TupleType,
    UnionType,
)


def _constant_type(node: IRConstant) -> Type:
    v = node.value
    if v is None:
        return NoneType

    if isinstance(v, bool):
        return BoolType

    if isinstance(v, int):
        return IntType

    if isinstance(v, float):
        return FloatType

    if isinstance(v, str):
        return StrType

    if isinstance(v, bytes):
        return BytesType

    if v is Ellipsis:
        return EllipsisType

    return AnyType


class AnnotationResolver:
    BUILTIN_TYPES: dict[str, Type] = {
        "int": IntType,
        "float": FloatType,
        "bool": BoolType,
        "str": StrType,
        "bytes": BytesType,
        "None": NoneType,
        "Any": AnyType,
        "list": ListType(AnyType),
        "dict": DictType(AnyType, AnyType),
        "tuple": TupleType(()),
        "set": SetType(AnyType),
        "Callable": CallableType((), AnyType),
    }

    GENERICS: dict[str, Callable] = {
        "list": lambda a: ListType(a[0]) if a else ListType(AnyType),
        "set": lambda a: SetType(a[0]) if a else SetType(AnyType),
        "tuple": lambda a: TupleType(tuple(a)) if a else TupleType(()),
        "dict": lambda a: (
            DictType(a[0], a[1]) if len(a) >= 2 else DictType(AnyType, AnyType)
        ),
        "Callable": lambda a: CallableType(
            tuple(a[0].items) if isinstance(a[0], TupleType) else (),
            a[1] if len(a) >= 2 else AnyType,
        ),
        "Union": lambda a: UnionType(tuple(a)),
        "Optional": lambda a: UnionType((a[0], NoneType)) if a else AnyType,
    }

    def __init__(self, symbol_table: SymbolTable, scope: Scope):
        self.symbol_table = symbol_table
        self.scope = scope

    def resolve(self, node: Optional[IRNode]) -> Type:
        if node is None:
            return AnyType
        return self._resolve(node)

    def _resolve(self, node: IRNode) -> Type:
        if isinstance(node, IRConstant):
            return _constant_type(node)

        if isinstance(node, (IRName, IRSymbolRef)):
            return self._name_type(node.name)

        if isinstance(node, IRSubscript):
            base = self._resolve(node.value)
            index = node.index
            args = (
                [self._resolve(e) for e in index.elements]
                if isinstance(index, IRTuple)
                else [self._resolve(index)]
            )

            if isinstance(base, ListType):
                return ListType(args[0]) if args else ListType(AnyType)

            if isinstance(base, SetType):
                return SetType(args[0]) if args else SetType(AnyType)

            if isinstance(base, DictType):
                return (
                    DictType(args[0], args[1])
                    if len(args) >= 2
                    else DictType(AnyType, AnyType)
                )

            if isinstance(base, TupleType):
                return TupleType(tuple(args))

            if isinstance(base, CallableType):
                if len(args) == 2:
                    param_args = args[0]
                    params = (
                        tuple(param_args.items)
                        if isinstance(param_args, TupleType)
                        else (param_args,)
                    )
                    return CallableType(params, args[1])

            if isinstance(node.value, (IRName, IRSymbolRef)):
                name = node.value.name
                if name in self.GENERICS:
                    return self.GENERICS[name](args)

            return AnyType

        if isinstance(node, IRTuple):
            return TupleType(tuple(self._resolve(e) for e in node.elements))

        if isinstance(node, IRCall):
            if isinstance(node.func, (IRName, IRSymbolRef)):
                name = node.func.name
                if name in ("Optional", "Union") and node.args:
                    args = [self._resolve(a) for a in node.args]
                    return self.GENERICS[name](args)

        return AnyType

    def _name_type(self, name: str) -> Type:
        if name in self.BUILTIN_TYPES:
            return self.BUILTIN_TYPES[name]

        sym = self._find_sym(name)
        if sym:
            if sym.kind == "class":
                return ClassType(name)

            if sym.kind == "function" and isinstance(sym.definition, IRFunctionDef):
                ptypes = [self.resolve(p.annotation) for p in sym.definition.params]
                rtype = self.resolve(sym.definition.returns)
                return CallableType(tuple(ptypes), rtype)

        return AnyType

    def _find_sym(self, name: str):
        for s in self.symbol_table.all():
            if s.name == name:
                return s

        return None

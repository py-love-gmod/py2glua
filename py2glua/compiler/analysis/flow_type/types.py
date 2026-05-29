from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class Type:
    def is_subtype(self, other: Type) -> bool:
        return other is AnyType or self._is_subtype_impl(other)

    def _is_subtype_impl(self, other: Type) -> bool:
        return False

    def __repr__(self) -> str:
        return type(self).__name__

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        return self._eq_impl(other)

    def __hash__(self):
        return hash(self._hash_tuple())

    def _eq_impl(self, other) -> bool:
        raise NotImplementedError

    def _hash_tuple(self) -> tuple:
        raise NotImplementedError


class _AnyType(Type):
    def _is_subtype_impl(self, other: Type) -> bool:
        return True

    def _eq_impl(self, other) -> bool:
        return True

    def _hash_tuple(self) -> tuple:
        return ()

    def __repr__(self) -> str:
        return "Any"


AnyType = _AnyType()


@dataclass(frozen=True)
class NominalType(Type):
    name: str

    def _is_subtype_impl(self, other: Type) -> bool:
        if isinstance(other, NominalType):
            return self.name == other.name
        return False

    def __repr__(self):
        return self.name

    def _eq_impl(self, other):
        return self.name == other.name

    def _hash_tuple(self):
        return (self.name,)


NoneType = NominalType("None")
BoolType = NominalType("bool")
IntType = NominalType("int")
FloatType = NominalType("float")
StrType = NominalType("str")
BytesType = NominalType("bytes")
EllipsisType = NominalType("Ellipsis")


@dataclass(frozen=True)
class UnionType(Type):
    types: tuple[Type, ...]

    def __post_init__(self):
        seen = set()
        uniq = []
        for t in self.types:
            if t not in seen:
                seen.add(t)
                uniq.append(t)

        uniq.sort(key=repr)
        object.__setattr__(self, "types", tuple(uniq))

    def _is_subtype_impl(self, other: Type) -> bool:
        return all(t.is_subtype(other) for t in self.types)

    def __repr__(self):
        return " | ".join(repr(t) for t in self.types)

    def _eq_impl(self, other):
        return set(self.types) == set(other.types)

    def _hash_tuple(self):
        return tuple(sorted(self.types, key=repr))


@dataclass(frozen=True)
class TupleType(Type):
    items: tuple[Type, ...]

    def _is_subtype_impl(self, other: Type) -> bool:
        if not isinstance(other, TupleType):
            return False
        return len(self.items) == len(other.items) and all(
            a.is_subtype(b) for a, b in zip(self.items, other.items)
        )

    def __repr__(self):
        return f"tuple[{', '.join(repr(i) for i in self.items)}]"

    def _eq_impl(self, other):
        return self.items == other.items

    def _hash_tuple(self):
        return self.items


@dataclass(frozen=True)
class ListType(Type):
    item: Type

    def _is_subtype_impl(self, other: Type) -> bool:
        if not isinstance(other, ListType):
            return False
        return self.item.is_subtype(other.item)

    def __repr__(self):
        return f"list[{repr(self.item)}]"

    def _eq_impl(self, other):
        return self.item == other.item

    def _hash_tuple(self):
        return (self.item,)


@dataclass(frozen=True)
class SetType(Type):
    item: Type

    def _is_subtype_impl(self, other: Type) -> bool:
        if not isinstance(other, SetType):
            return False

        return self.item.is_subtype(other.item)

    def __repr__(self):
        return f"set[{repr(self.item)}]"

    def _eq_impl(self, other):
        return self.item == other.item

    def _hash_tuple(self):
        return (self.item,)


@dataclass(frozen=True)
class DictType(Type):
    key: Type
    value: Type

    def _is_subtype_impl(self, other: Type) -> bool:
        if not isinstance(other, DictType):
            return False

        return self.key.is_subtype(other.key) and self.value.is_subtype(other.value)

    def __repr__(self):
        return f"dict[{repr(self.key)}, {repr(self.value)}]"

    def _eq_impl(self, other):
        return self.key == other.key and self.value == other.value

    def _hash_tuple(self):
        return (self.key, self.value)


@dataclass(frozen=True)
class CallableType(Type):
    params: tuple[Type, ...]
    return_type: Type

    def _is_subtype_impl(self, other: Type) -> bool:
        if not isinstance(other, CallableType):
            return False

        if len(self.params) != len(other.params):
            return False

        return all(
            p.is_subtype(op) for p, op in zip(self.params, other.params)
        ) and self.return_type.is_subtype(other.return_type)

    def __repr__(self):
        args = ", ".join(repr(p) for p in self.params)
        return f"({args}) -> {repr(self.return_type)}"

    def _eq_impl(self, other):
        return self.params == other.params and self.return_type == other.return_type

    def _hash_tuple(self):
        return (self.params, self.return_type)


@dataclass(frozen=True)
class ClassType(Type):
    name: str
    fields: dict[str, Type] = field(default_factory=dict)
    base: Optional[ClassType] = None

    def _is_subtype_impl(self, other: Type) -> bool:
        if not isinstance(other, ClassType):
            return False

        return self.name == other.name or (
            self.base is not None and self.base.is_subtype(other)
        )

    def get_field(self, name: str) -> Type:
        return self.fields.get(name) or (
            self.base.get_field(name) if self.base else AnyType
        )

    def __repr__(self):
        return f"class {self.name}"

    def _eq_impl(self, other):
        return (
            self.name == other.name
            and self.fields == other.fields
            and self.base == other.base
        )

    def _hash_tuple(self):
        return (self.name, frozenset(self.fields.items()), self.base)


@dataclass(frozen=True)
class ModuleType(Type):
    name: str
    members: dict[str, Type] = field(default_factory=dict)

    def _is_subtype_impl(self, other: Type) -> bool:
        return isinstance(other, ModuleType)

    def __repr__(self):
        return f"module {self.name}"

    def _eq_impl(self, other):
        return self.name == other.name and self.members == other.members

    def _hash_tuple(self):
        return (self.name, frozenset(self.members.items()))

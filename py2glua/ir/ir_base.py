from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from ..glua.realm import Realm


# region БАЗОВЫЙ КЛАСС
@dataclass
class IRNode:
    lineno: int | None
    col_offset: int | None

    file: File | None
    parent: IRNode | None

    def walk(self) -> Iterator[IRNode]:
        raise NotImplementedError


# endregion


# region ФАЙЛ
@dataclass
class File(IRNode):
    path: Path | None = None
    realm: Realm | None = None
    meta_tags: list[str] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)

    meta: dict = field(default_factory=dict)

    def walk(self) -> Iterator[IRNode]:
        yield self
        for node in self.body:
            yield from node.walk()


# endregion


# region ИМПОРТЫ
class ImportType(Enum):
    UNKNOWN = auto()
    PYTHON_STD = auto()
    EXTERNAL = auto()
    LOCAL = auto()
    INTERNAL = auto()


@dataclass
class Import(IRNode):
    module: str | None = None
    names: list[str] = field(default_factory=list)
    aliases: list[str | None] = field(default_factory=list)
    import_type: ImportType = ImportType.UNKNOWN

    def walk(self) -> Iterator[IRNode]:
        yield self


# endregion


# region ПЕРЕМЕННЫЕ И КОНСТАНТЫ
@dataclass
class Constant(IRNode):
    value: object

    def walk(self) -> Iterator[IRNode]:
        yield self


@dataclass
class VarLoad(IRNode):
    name: str

    def walk(self) -> Iterator[IRNode]:
        yield self


@dataclass
class VarStore(IRNode):
    name: str
    value: IRNode
    annotation: str | None = None

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.value.walk()


# endregion


# region ВЫРАЖЕНИЯ И ВЫЗОВЫ
@dataclass
class Call(IRNode):
    func: IRNode
    args: list[IRNode] = field(default_factory=list)
    kwargs: dict[str, IRNode] = field(default_factory=dict)

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.func.walk()
        for arg in self.args:
            yield from arg.walk()

        for kw_value in self.kwargs.values():
            yield from kw_value.walk()


@dataclass
class Attribute(IRNode):
    value: IRNode
    attr: str  # имя поля

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.value.walk()


@dataclass
class Subscript(IRNode):
    value: IRNode  # таблица / список
    index: IRNode  # индекс

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.value.walk()
        yield from self.index.walk()


# endregion


# region УНАРНЫЕ, БИНАРНЫЕ, БУЛЕВЫЕ ОПЕРАЦИИ
class UnaryOpType(Enum):
    NEG = auto()  # -x
    POS = auto()  # +x
    NOT = auto()  # not x
    BIT_NOT = auto()  # ~x


@dataclass
class UnaryOp(IRNode):
    op: UnaryOpType
    operand: IRNode

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.operand.walk()


class BinOpType(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    BIT_OR = auto()
    BIT_AND = auto()
    BIT_XOR = auto()
    BIT_LSHIFT = auto()
    BIT_RSHIFT = auto()
    FLOOR_DIV = auto()
    POW = auto()


@dataclass
class BinOp(IRNode):
    op: BinOpType
    left: IRNode
    right: IRNode

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.left.walk()
        yield from self.right.walk()


class BoolOpType(Enum):
    AND = auto()
    OR = auto()
    NOT = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    IS = auto()
    IS_NOT = auto()
    IN = auto()
    NOT_IN = auto()


@dataclass
class BoolOp(IRNode):
    op: BoolOpType
    left: IRNode | None = None
    right: IRNode | None = None

    def walk(self) -> Iterator[IRNode]:
        yield self
        if self.left:
            yield from self.left.walk()

        if self.right:
            yield from self.right.walk()


@dataclass
class Compare(IRNode):
    op: BoolOpType
    left: IRNode
    right: IRNode

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.left.walk()
        yield from self.right.walk()


# endregion


# region ФУНКЦИИ
@dataclass
class FunctionDef(IRNode):
    name: str
    args: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        for node in self.body:
            yield from node.walk()


@dataclass
class Return(IRNode):
    value: IRNode | None = None

    def walk(self) -> Iterator[IRNode]:
        yield self
        if self.value:
            yield from self.value.walk()


# endregion


# region КЛАССЫ
@dataclass
class ClassDef(IRNode):
    name: str
    bases: list[IRNode] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        for base in self.bases:
            yield from base.walk()

        for node in self.body:
            yield from node.walk()


# endregion


# region КОНТРОЛЬ ПОТОКА
@dataclass
class If(IRNode):
    test: IRNode
    body: list[IRNode] = field(default_factory=list)
    orelse: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.test.walk()
        for node in self.body:
            yield from node.walk()

        for node in self.orelse:
            yield from node.walk()


@dataclass
class While(IRNode):
    test: IRNode
    body: list[IRNode] = field(default_factory=list)
    orelse: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.test.walk()
        for node in self.body:
            yield from node.walk()

        for node in self.orelse:
            yield from node.walk()


@dataclass
class For(IRNode):
    target: IRNode
    iter: IRNode
    body: list[IRNode] = field(default_factory=list)
    orelse: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.target.walk()
        yield from self.iter.walk()
        for node in self.body:
            yield from node.walk()

        for node in self.orelse:
            yield from node.walk()


@dataclass
class With(IRNode):
    context: IRNode
    target: IRNode | None
    body: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.context.walk()
        if self.target:
            yield from self.target.walk()

        for node in self.body:
            yield from node.walk()


@dataclass
class ExceptHandler(IRNode):
    type: IRNode | None
    name: str | None
    body: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        if self.type:
            yield from self.type.walk()

        for node in self.body:
            yield from node.walk()


@dataclass
class Try(IRNode):
    body: list[IRNode] = field(default_factory=list)
    handlers: list[ExceptHandler] = field(default_factory=list)
    orelse: list[IRNode] = field(default_factory=list)
    finalbody: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        for node in self.body:
            yield from node.walk()

        for h in self.handlers:
            yield from h.walk()

        for node in self.orelse:
            yield from node.walk()

        for node in self.finalbody:
            yield from node.walk()


@dataclass
class Break(IRNode):
    def walk(self) -> Iterator[IRNode]:
        yield self


@dataclass
class Continue(IRNode):
    def walk(self) -> Iterator[IRNode]:
        yield self


@dataclass
class Pass(IRNode):
    def walk(self) -> Iterator[IRNode]:
        yield self


# endregion


# region КОНТЕЙНЕРЫ (СПИСКИ / КОРТЕЖИ / СЛОВАРИ)
@dataclass
class ListLiteral(IRNode):
    elements: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        for elem in self.elements:
            yield from elem.walk()


@dataclass
class TupleLiteral(IRNode):
    elements: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        for elem in self.elements:
            yield from elem.walk()


@dataclass
class DictLiteral(IRNode):
    keys: list[IRNode] = field(default_factory=list)
    values: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        for key in self.keys:
            yield from key.walk()

        for val in self.values:
            yield from val.walk()


# endregion


# region LAMBDA
@dataclass
class Lambda(IRNode):
    args: list[str] = field(default_factory=list)
    body: IRNode | None = None  # expression

    def walk(self) -> Iterator[IRNode]:
        yield self
        if self.body:
            yield from self.body.walk()


# endregion


# region WALRUS (NamedExpr)
@dataclass
class NamedExpr(IRNode):
    name: str
    value: IRNode

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.value.walk()


# endregion


# region F-STRINGS
@dataclass
class FStringPart(IRNode):
    def walk(self) -> Iterator[IRNode]:
        yield self


@dataclass
class FStringText(FStringPart):
    text: str = ""

    def walk(self) -> Iterator[IRNode]:
        yield self


@dataclass
class FStringExpr(FStringPart):
    value: IRNode
    conversion: str | None = None
    format_spec: str | None = None

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.value.walk()


@dataclass
class FString(IRNode):
    parts: list[FStringPart] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        for p in self.parts:
            yield from p.walk()


# endregion


# region COMPREHENSIONS
@dataclass
class ComprehensionFor(IRNode):
    target: IRNode
    iter: IRNode
    ifs: list[IRNode] = field(default_factory=list)
    is_async: bool = False

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.target.walk()
        yield from self.iter.walk()
        for cond in self.ifs:
            yield from cond.walk()


@dataclass
class ListComp(IRNode):
    elt: IRNode
    generators: list[ComprehensionFor] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.elt.walk()
        for gen in self.generators:
            yield from gen.walk()


# endregion


# region MATCH/CASE (PEP 634)
class PatternKind(Enum):
    WILDCARD = auto()  # _
    VALUE = auto()  # literal or dotted attr
    NAME = auto()  # capture name
    OR = auto()  # OR of subpatterns
    SEQUENCE = auto()  # [p1, p2, ...] or (..)
    MAPPING = auto()  # {k1: p1, ...}


@dataclass
class Pattern(IRNode):
    kind: PatternKind
    value: IRNode | None = None
    name: str | None = None
    patterns: list["Pattern"] = field(default_factory=list)
    keys: list[IRNode] = field(default_factory=list)
    values: list["Pattern"] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        if self.value:
            yield from self.value.walk()

        for p in self.patterns:
            yield from p.walk()

        for k in self.keys:
            yield from k.walk()

        for v in self.values:
            yield from v.walk()


@dataclass
class MatchCase(IRNode):
    pattern: Pattern
    guard: IRNode | None = None
    body: list[IRNode] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.pattern.walk()
        if self.guard:
            yield from self.guard.walk()

        for n in self.body:
            yield from n.walk()


@dataclass
class Match(IRNode):
    subject: IRNode
    cases: list[MatchCase] = field(default_factory=list)

    def walk(self) -> Iterator[IRNode]:
        yield self
        yield from self.subject.walk()
        for c in self.cases:
            yield from c.walk()


# endregion

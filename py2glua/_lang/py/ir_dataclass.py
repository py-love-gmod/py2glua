from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path

from ...glua.core.types import nil


# Base
@dataclass
class PyIRNode:
    line: int | None
    offset: int | None

    def walk(self):
        raise NotImplementedError()


# File / Decorators
@dataclass
class PyIRFile(PyIRNode):
    path: Path | None
    body: list["PyIRNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for node in self.body:
            yield from node.walk()

    def __hash__(self) -> int:
        return hash(str(self.path))


@dataclass
class PyIRDecorator(PyIRNode):
    name: str
    args_p: list["PyIRNode"] = field(default_factory=list)
    args_kw: dict[str, "PyIRNode"] = field(default_factory=dict)

    def walk(self):
        yield self
        for a in self.args_p:
            yield from a.walk()
        for a in self.args_kw.values():
            yield from a.walk()


# Import
class PyIRImportType(IntEnum):
    UNKNOWN = auto()
    LOCAL = auto()
    STD_LIB = auto()
    INTERNAL = auto()
    EXTERNAL = auto()


@dataclass
class PyIRImport(PyIRNode):
    modules: list[str]
    names: list[str | tuple[str, str]]
    if_from: bool
    level: int
    itype: PyIRImportType = PyIRImportType.UNKNOWN

    def walk(self):
        yield self


# Constants / Names
@dataclass
class PyIRConstant(PyIRNode):
    value: object | nil

    def walk(self):
        yield self


@dataclass
class PyIRVarUse(PyIRNode):
    name: str

    def walk(self):
        yield self


@dataclass
class PyIRVarCreate(PyIRNode):
    name: str
    is_global: bool = False

    def walk(self):
        yield self


@dataclass
class PyIRFString(PyIRNode):
    parts: list["PyIRNode | str"]

    def walk(self):
        yield self
        for p in self.parts:
            if isinstance(p, PyIRNode):
                yield from p.walk()


# Annotation
@dataclass
class PyIRAnnotation(PyIRNode):
    name: str
    annotation: str

    def walk(self):
        yield self


@dataclass
class PyIRAnnotatedAssign(PyIRNode):
    name: str
    annotation: str
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.value.walk()


# Attribute / Subscript / Collections
@dataclass
class PyIRAttribute(PyIRNode):
    value: PyIRNode
    attr: str

    def walk(self):
        yield self
        yield from self.value.walk()


@dataclass
class PyIRSubscript(PyIRNode):
    value: PyIRNode
    index: PyIRNode

    def walk(self):
        yield self
        yield from self.value.walk()
        yield from self.index.walk()


@dataclass
class PyIRList(PyIRNode):
    elements: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for e in self.elements:
            yield from e.walk()


@dataclass
class PyIRTuple(PyIRNode):
    elements: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for e in self.elements:
            yield from e.walk()


@dataclass
class PyIRSet(PyIRNode):
    elements: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for e in self.elements:
            yield from e.walk()


@dataclass
class PyIRDictItem(PyIRNode):
    key: PyIRNode
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.key.walk()
        yield from self.value.walk()


@dataclass
class PyIRDict(PyIRNode):
    items: list[PyIRDictItem] = field(default_factory=list)

    def walk(self):
        yield self
        for i in self.items:
            yield from i.walk()


# Operations
class PyBinOPType(IntEnum):
    OR = auto()
    AND = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    IN = auto()
    NOT_IN = auto()
    IS = auto()
    IS_NOT = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_AND = auto()
    BIT_LSHIFT = auto()
    BIT_RSHIFT = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOORDIV = auto()
    MOD = auto()
    POW = auto()


class PyUnaryOPType(IntEnum):
    PLUS = auto()
    MINUS = auto()
    NOT = auto()
    BIT_INV = auto()


@dataclass
class PyIRBinOP(PyIRNode):
    op: PyBinOPType
    left: PyIRNode
    right: PyIRNode

    def walk(self):
        yield self
        yield from self.left.walk()
        yield from self.right.walk()


@dataclass
class PyIRUnaryOP(PyIRNode):
    op: PyUnaryOPType
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.value.walk()


# Assignment
class PyAugAssignType(IntEnum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOORDIV = auto()
    MOD = auto()
    POW = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_AND = auto()
    LSHIFT = auto()
    RSHIFT = auto()


@dataclass
class PyIRAssign(PyIRNode):
    targets: list[PyIRNode]
    value: PyIRNode

    def walk(self):
        yield self
        for t in self.targets:
            yield from t.walk()

        yield from self.value.walk()


@dataclass
class PyIRAugAssign(PyIRNode):
    target: PyIRNode
    op: PyAugAssignType
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.target.walk()
        yield from self.value.walk()


# Call / Function / Class
@dataclass
class PyIRCall(PyIRNode):
    func: PyIRNode
    args_p: list[PyIRNode] = field(default_factory=list)
    args_kw: dict[str, PyIRNode] = field(default_factory=dict)

    def walk(self):
        yield self
        yield from self.func.walk()
        for a in self.args_p:
            yield from a.walk()

        for a in self.args_kw.values():
            yield from a.walk()


@dataclass
class PyIRFunctionDef(PyIRNode):
    name: str
    signature: dict[str, tuple[str | None, str | None]]
    decorators: list[PyIRDecorator] = field(default_factory=list)
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for d in self.decorators:
            yield from d.walk()

        for n in self.body:
            yield from n.walk()


@dataclass
class PyIRClassDef(PyIRNode):
    name: str
    decorators: list[PyIRDecorator] = field(default_factory=list)
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for d in self.decorators:
            yield from d.walk()

        for n in self.body:
            yield from n.walk()


# Control Flow
@dataclass
class PyIRIf(PyIRNode):
    test: PyIRNode
    body: list[PyIRNode] = field(default_factory=list)
    orelse: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.test.walk()
        for n in self.body:
            yield from n.walk()

        for n in self.orelse:
            yield from n.walk()


@dataclass
class PyIRWhile(PyIRNode):
    test: PyIRNode
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.test.walk()
        for n in self.body:
            yield from n.walk()


@dataclass
class PyIRFor(PyIRNode):
    target: PyIRNode
    iter: PyIRNode
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.target.walk()
        yield from self.iter.walk()
        for n in self.body:
            yield from n.walk()


@dataclass
class PyIRReturn(PyIRNode):
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.value.walk()


@dataclass
class PyIRBreak(PyIRNode):
    def walk(self):
        yield self


@dataclass
class PyIRContinue(PyIRNode):
    def walk(self):
        yield self


# With
@dataclass
class PyIRWithItem(PyIRNode):
    context_expr: PyIRNode
    optional_vars: PyIRNode | None = None

    def walk(self):
        yield self
        yield from self.context_expr.walk()
        if self.optional_vars is not None:
            yield from self.optional_vars.walk()


@dataclass
class PyIRWith(PyIRNode):
    items: list[PyIRWithItem] = field(default_factory=list)
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for item in self.items:
            yield from item.walk()

        for node in self.body:
            yield from node.walk()


# Comment
@dataclass
class PyIRComment(PyIRNode):
    value: str

    def walk(self):
        yield self


# Pass
@dataclass
class PyIRPass(PyIRNode):
    def walk(self):
        yield self


# Backend escape hatch
class PyIRBackendKind(IntEnum):
    GLOBAL = auto()
    CALL = auto()
    ATTR = auto()
    INDEX = auto()
    RAW = auto()


@dataclass
class PyIRBackendExpr(PyIRNode):
    kind: PyIRBackendKind
    name: str
    args_p: list[PyIRNode] = field(default_factory=list)
    args_kw: dict[str, PyIRNode] = field(default_factory=dict)

    def walk(self):
        yield self
        for a in self.args_p:
            yield from a.walk()

        for a in self.args_kw.values():
            yield from a.walk()

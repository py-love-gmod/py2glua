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
    imports: list[Path] = field(default_factory=list)

    def walk(self):
        yield self
        for node in self.body:
            yield from node.walk()

    def __hash__(self) -> int:
        return hash(str(self.path))


@dataclass
class PyIRDecorator(PyIRNode):
    exper: PyIRNode
    args_p: list["PyIRNode"] = field(default_factory=list)
    args_kw: dict[str, "PyIRNode"] = field(default_factory=dict)

    def walk(self):
        yield self

        yield from self.exper.walk()

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

    def __str__(self) -> str:
        prefix = "." * self.level if self.level > 0 else ""

        if self.if_from:
            mods = prefix + ".".join(self.modules)

            if not self.names:
                return f"-- <PYTHON from {mods} import * >"

            parts: list[str] = []
            for n in self.names:
                if isinstance(n, tuple):
                    src, alias = n
                    parts.append(f"{src} as {alias}")
                else:
                    parts.append(n)

            return f"-- <PYTHON from {mods} import {', '.join(parts)} >"

        mods = prefix + ".".join(self.modules)
        return f"-- <PYTHON import {mods} >"


# Constants / Names
@dataclass
class PyIRConstant(PyIRNode):
    value: object | nil

    def walk(self):
        yield self

    def __str__(self) -> str:
        v = self.value

        if v is nil:
            return "nil"

        if v is None:
            return "TODO: None"

        if isinstance(v, bool):
            return "true" if v else "false"

        if isinstance(v, str):
            return repr(v)

        if isinstance(v, (int, float)):
            return str(v)

        raise AssertionError(
            "Неподдерживаемый тип константы в PyIRConstant.\n"
            f"Тип: {type(v).__name__}\n"
            f"Значение: {v!r}\n"
            f"Позиция: line={self.line}, offset={self.offset}"
        )


@dataclass
class PyIRVarUse(PyIRNode):
    name: str

    def walk(self):
        yield self

    def __str__(self) -> str:
        return self.name


@dataclass
class PyIRVarCreate(PyIRNode):
    """
    Statement-only node.

    Contract:
    - Must not appear inside expressions.
    - Emit uses `local <name>` unless is_global=True.
    """

    name: str
    is_global: bool = False

    def walk(self):
        yield self

    def __str__(self) -> str:
        return self.name if self.is_global else f"local {self.name}"


# FString (must be lowered)
@dataclass
class PyIRFString(PyIRNode):
    parts: list["PyIRNode | str"]

    def walk(self):
        yield self
        for p in self.parts:
            if isinstance(p, PyIRNode):
                yield from p.walk()

    def __str__(self) -> str:
        raise AssertionError("PyIRFString must be lowered before emit")


# Annotation (Python-only; must be stripped)
@dataclass
class PyIRAnnotation(PyIRVarCreate):
    """
    Python-only annotation-only statement: `x: T`
    Must be stripped before emit.
    """

    annotation: str = ""

    def walk(self):
        yield self

    def __str__(self) -> str:
        # If this reaches emit - it's a pipeline bug.
        raise AssertionError("PyIRAnnotation must be stripped before emit")


@dataclass
class PyIRAnnotatedAssign(PyIRVarCreate):
    """
    Python-only annotated assignment: `x: T = expr`
    Must be lowered/stripped before emit (depending on your design).
    """

    annotation: str = ""
    value: PyIRNode = field(default_factory=lambda: PyIRPass(line=None, offset=None))

    def walk(self):
        yield self
        yield from self.value.walk()

    def __str__(self) -> str:
        raise AssertionError("PyIRAnnotatedAssign must be lowered before emit")


# Attribute / Subscript / Collections
@dataclass
class PyIRAttribute(PyIRNode):
    value: PyIRNode
    attr: str

    def walk(self):
        yield self
        yield from self.value.walk()

    def __str__(self) -> str:
        return f"{self.value}.{self.attr}"


@dataclass
class PyIRSubscript(PyIRNode):
    value: PyIRNode
    index: PyIRNode

    def walk(self):
        yield self
        yield from self.value.walk()
        yield from self.index.walk()

    def __str__(self) -> str:
        return f"{self.value}[{self.index}]"


@dataclass
class PyIRList(PyIRNode):
    elements: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for e in self.elements:
            yield from e.walk()

    def __str__(self) -> str:
        raise AssertionError("PyIRList must be lowered before emit")


@dataclass
class PyIRTuple(PyIRNode):
    elements: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for e in self.elements:
            yield from e.walk()

    def __str__(self) -> str:
        raise AssertionError("PyIRTuple must be lowered before emit")


@dataclass
class PyIRSet(PyIRNode):
    elements: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for e in self.elements:
            yield from e.walk()

    def __str__(self) -> str:
        raise AssertionError("PyIRSet must be lowered before emit")


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

    def __str__(self) -> str:
        raise AssertionError("PyIRDict must be lowered before emit")


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

    def __str__(self) -> str:
        raise AssertionError("PyIRBinOP must be lowered before emit")


@dataclass
class PyIRUnaryOP(PyIRNode):
    op: PyUnaryOPType
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.value.walk()

    def __str__(self) -> str:
        raise AssertionError("PyIRUnaryOP must be lowered before emit")


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

    def __str__(self) -> str:
        raise AssertionError("PyIRAssign must be emitted by a statement emitter")


@dataclass
class PyIRAugAssign(PyIRNode):
    target: PyIRNode
    op: PyAugAssignType
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.target.walk()
        yield from self.value.walk()

    def __str__(self) -> str:
        raise AssertionError("PyIRAugAssign must be lowered before emit")


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

    def __str__(self) -> str:
        raise AssertionError("PyIRCall must be lowered before emit")


@dataclass
class PyIRFunctionDef(PyIRNode):
    name: str
    signature: dict[str, tuple[str | None, str | None]]
    returns: str | None
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
    bases: list[PyIRNode] = field(default_factory=list)
    decorators: list[PyIRDecorator] = field(default_factory=list)
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for b in self.bases:
            yield from b.walk()

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


# Comment / Pass
@dataclass
class PyIRComment(PyIRNode):
    value: str

    def walk(self):
        yield self


@dataclass
class PyIRPass(PyIRNode):
    def walk(self):
        yield self


# Emit escape hatch
class PyIREmitKind(IntEnum):
    GLOBAL = auto()
    CALL = auto()
    ATTR = auto()
    INDEX = auto()
    RAW = auto()


@dataclass
class PyIREmitExpr(PyIRNode):
    kind: PyIREmitKind
    name: str
    args_p: list[PyIRNode] = field(default_factory=list)
    args_kw: dict[str, PyIRNode] = field(default_factory=dict)

    def walk(self):
        yield self
        for a in self.args_p:
            yield from a.walk()

        for a in self.args_kw.values():
            yield from a.walk()

    def __str__(self) -> str:
        if self.kind == PyIREmitKind.GLOBAL:
            return self.name

        if self.kind == PyIREmitKind.RAW:
            return self.name

        if self.kind == PyIREmitKind.ATTR:
            if len(self.args_p) != 1:
                raise AssertionError("ATTR expects exactly 1 positional arg (base)")

            return f"{self.args_p[0]}.{self.name}"

        if self.kind == PyIREmitKind.INDEX:
            if len(self.args_p) != 2:
                raise AssertionError(
                    "INDEX expects exactly 2 positional args (base, index)"
                )

            return f"{self.args_p[0]}[{self.args_p[1]}]"

        if self.kind == PyIREmitKind.CALL:
            args = ", ".join(str(a) for a in self.args_p)
            if self.args_kw:
                raise AssertionError("CALL does not support keyword args in emit expr")

            return f"{self.name}({args})"

        raise AssertionError("unknown emit kind")

from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Any


# region Base
@dataclass
class PyIRNode:
    line: int | None
    offset: int | None

    def walk(self):
        raise NotImplementedError()


# endregion


# region Context
@dataclass
class PyIRContext:
    parent_context: "PyIRContext | None" = None
    meta: dict[str, Any] = field(default_factory=dict)
    scope_name: set[str] = field(default_factory=set)


# endregion


# region File / Decorator
@dataclass
class PyIRFile(PyIRNode):
    path: Path | None
    context: PyIRContext
    body: list["PyIRNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for node in self.body:
            yield from node.walk()


@dataclass
class PyIRDecorator(PyIRNode):
    name: str
    args_p: list[PyIRNode] = field(default_factory=list)
    args_kw: dict[str, PyIRNode] = field(default_factory=dict)

    def walk(self):
        yield self
        for arg_p in self.args_p:
            yield from arg_p.walk()

        for arg_kw in self.args_kw.values():
            yield from arg_kw.walk()


# endregion


# region Import
class PyIRImportType(IntEnum):
    UNKNOWN = auto()
    LOCAL = auto()
    STD_LIB = auto()
    INTERNAL = auto()
    EXTERNAL = auto()


@dataclass
class PyIRImport(PyIRNode):
    modules: list[str]
    i_type: PyIRImportType = PyIRImportType.UNKNOWN

    def walk(self):
        yield self


# endregion


# region Vars & Constants
@dataclass
class PyIRConstant(PyIRNode):
    value: object

    def walk(self):
        yield self


@dataclass
class PyIRVarCreate(PyIRNode):
    name: str
    is_global: bool = False

    def walk(self):
        yield self


@dataclass
class PyIRVarUse(PyIRNode):
    name: str

    def walk(self):
        yield self


# endregion


# region Attribute / Subscript / Collections
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
        for el in self.elements:
            yield from el.walk()


@dataclass
class PyIRTuple(PyIRNode):
    elements: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for el in self.elements:
            yield from el.walk()


@dataclass
class PyIRSet(PyIRNode):
    elements: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for el in self.elements:
            yield from el.walk()


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
        for item in self.items:
            yield from item.walk()


# endregion


# region OP
class PyBinOPType(IntEnum):
    OR = auto()  # or
    AND = auto()  # and

    EQ = auto()  # ==
    NE = auto()  # !=
    LT = auto()  # <
    GT = auto()  # >
    LE = auto()  # <=
    GE = auto()  # >=

    IN = auto()  # in
    NOT_IN = auto()  # not in

    IS = auto()  # is
    IS_NOT = auto()  # is not

    BIT_OR = auto()  # |
    BIT_XOR = auto()  # ^
    BIT_AND = auto()  # &
    BIT_LSHIFT = auto()  # <<
    BIT_RSHIFT = auto()  # >>

    ADD = auto()  # +
    SUB = auto()  # -

    MUL = auto()  # *
    DIV = auto()  # /
    FLOORDIV = auto()  # //
    MOD = auto()  # %

    POW = auto()  # **


class PyUnaryOPType(IntEnum):
    PLUS = auto()  # +
    MINUS = auto()  # -

    NOT = auto()  # not

    BIT_INV = auto()  # ~


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


# endregion


# region Assignment
class PyAugAssignType(IntEnum):
    ADD = auto()  # +=
    SUB = auto()  # -=
    MUL = auto()  # *=
    DIV = auto()  # /=
    FLOORDIV = auto()  # //=
    MOD = auto()  # %=
    POW = auto()  # **=
    BIT_OR = auto()  # |=
    BIT_XOR = auto()  # ^=
    BIT_AND = auto()  # &=
    LSHIFT = auto()  # <<=
    RSHIFT = auto()  # >>=


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


# endregion


# region Function
@dataclass
class PyIRCall(PyIRNode):
    name: str
    args_p: list[PyIRNode] = field(default_factory=list)
    args_kw: dict[str, PyIRNode] = field(default_factory=dict)

    def walk(self):
        yield self
        for arg_p in self.args_p:
            yield from arg_p.walk()

        for arg_kw in self.args_kw.values():
            yield from arg_kw.walk()


@dataclass
class PyIRFunctionDef(PyIRNode):
    name: str
    signature: dict[str, str | int]
    context: PyIRContext
    decorators: list[PyIRDecorator] = field(default_factory=list)
    body: list["PyIRNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for dec in self.decorators:
            yield from dec.walk()

        for node in self.body:
            yield from node.walk()


# endregion


# region Class
@dataclass
class PyIRClassDef(PyIRNode):
    name: str
    context: PyIRContext
    decorators: list[PyIRDecorator] = field(default_factory=list)
    body: list["PyIRNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for dec in self.decorators:
            yield from dec.walk()

        for node in self.body:
            yield from node.walk()


# endregion


# region Del
@dataclass
class PyIRDel(PyIRNode):
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.value.walk()


# endregion


# region Return
@dataclass
class PyIRReturn(PyIRNode):
    value: PyIRNode

    def walk(self):
        yield self
        yield from self.value.walk()


# endregion


# region Control Flow: If / Loop / Break / Continue
@dataclass
class PyIRIf(PyIRNode):
    test: PyIRNode
    body: list[PyIRNode] = field(default_factory=list)
    orelse: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.test.walk()

        for node in self.body:
            yield from node.walk()

        for node in self.orelse:
            yield from node.walk()


@dataclass
class PyIRWhile(PyIRNode):
    test: PyIRNode
    body: list[PyIRNode] = field(default_factory=list)
    orelse: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.test.walk()

        for node in self.body:
            yield from node.walk()

        for node in self.orelse:
            yield from node.walk()


@dataclass
class PyIRFor(PyIRNode):
    target: PyIRNode
    iter: PyIRNode
    body: list[PyIRNode] = field(default_factory=list)
    orelse: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.target.walk()
        yield from self.iter.walk()

        for node in self.body:
            yield from node.walk()

        for node in self.orelse:
            yield from node.walk()


@dataclass
class PyIRBreak(PyIRNode):
    def walk(self):
        yield self


@dataclass
class PyIRContinue(PyIRNode):
    def walk(self):
        yield self


# endregion


# region Try / Except / With
@dataclass
class PyIRExceptHandler(PyIRNode):
    type: PyIRNode | None  # except ValueError:
    name: str | None  # except ValueError as e:
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        if self.type is not None:
            yield from self.type.walk()

        for node in self.body:
            yield from node.walk()


@dataclass
class PyIRTry(PyIRNode):
    body: list[PyIRNode] = field(default_factory=list)
    handlers: list[PyIRExceptHandler] = field(default_factory=list)
    orelse: list[PyIRNode] = field(default_factory=list)
    finalbody: list[PyIRNode] = field(default_factory=list)

    def walk(self):
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


# endregion


# region Comment
@dataclass
class PyIRComment(PyIRNode):
    value: str

    def walk(self):
        yield self


# endregion


# region Pass
@dataclass
class PyIRPass(PyIRNode):
    def walk(self):
        yield self


# endregion

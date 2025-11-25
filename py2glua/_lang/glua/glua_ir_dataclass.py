from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path


# region Base
@dataclass
class GluaNode:
    line: int | None
    offset: int | None

    def walk(self):
        raise NotImplementedError()


# endregion


# region File / Chunk
@dataclass
class GluaFile(GluaNode):
    path: Path | None
    body: list["GluaNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for node in self.body:
            yield from node.walk()


# endregion


# region Constants / Names / Basic expressions
@dataclass
class GluaConstant(GluaNode):
    value: object

    def walk(self):
        yield self


@dataclass
class GluaNil(GluaNode):
    def walk(self):
        yield self


@dataclass
class GluaName(GluaNode):
    name: str

    def walk(self):
        yield self


@dataclass
class GluaVarArg(GluaNode):
    def walk(self):
        yield self


@dataclass
class GluaAttribute(GluaNode):
    value: GluaNode
    attr: str

    def walk(self):
        yield self
        yield from self.value.walk()


@dataclass
class GluaIndex(GluaNode):
    value: GluaNode
    index: GluaNode

    def walk(self):
        yield self
        yield from self.value.walk()
        yield from self.index.walk()


# endregion


# region Tables
@dataclass
class GluaTableField(GluaNode):
    key: GluaNode | None
    value: GluaNode

    def walk(self):
        yield self
        if self.key is not None:
            yield from self.key.walk()

        yield from self.value.walk()


@dataclass
class GluaTable(GluaNode):
    fields: list[GluaTableField] = field(default_factory=list)

    def walk(self):
        yield self
        for f in self.fields:
            yield from f.walk()


# endregion


# region Operators
class GluaBinOpType(IntEnum):
    OR = auto()  # or
    AND = auto()  # and

    EQ = auto()  # ==
    NE = auto()  # ~=
    LT = auto()  # <
    GT = auto()  # >
    LE = auto()  # <=
    GE = auto()  # >=

    CONCAT = auto()  # ..

    ADD = auto()  # +
    SUB = auto()  # -
    MUL = auto()  # *
    DIV = auto()  # /
    MOD = auto()  # %
    POW = auto()  # ^


class GluaUnaryOpType(IntEnum):
    MINUS = auto()  # -expr
    NOT = auto()  # not expr
    LEN = auto()  # #expr


@dataclass
class GluaBinOp(GluaNode):
    op: GluaBinOpType
    left: GluaNode
    right: GluaNode

    def walk(self):
        yield self
        yield from self.left.walk()
        yield from self.right.walk()


@dataclass
class GluaUnaryOp(GluaNode):
    op: GluaUnaryOpType
    value: GluaNode

    def walk(self):
        yield self
        yield from self.value.walk()


# endregion


# region Call / Expression-as-statement
@dataclass
class GluaCall(GluaNode):
    func: GluaNode
    args: list[GluaNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.func.walk()

        for a in self.args:
            yield from a.walk()


# endregion


# region Assignment
@dataclass
class GluaAssign(GluaNode):
    targets: list[GluaNode]
    values: list[GluaNode] = field(default_factory=list)
    is_local: bool = False

    def walk(self):
        yield self
        for t in self.targets:
            yield from t.walk()

        for v in self.values:
            yield from v.walk()


# endregion


# region Control Flow: If / While / For / Return / Break / Continue
@dataclass
class GluaIf(GluaNode):
    """
    if test then body end

    elif/else рекомендуется разворачивать в цепочку вложенных GluaIf,
    как ты уже делаешь в PyIRIf.
    """

    test: GluaNode
    body: list[GluaNode] = field(default_factory=list)
    orelse: list[GluaNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.test.walk()

        for node in self.body:
            yield from node.walk()

        for node in self.orelse:
            yield from node.walk()


@dataclass
class GluaWhile(GluaNode):
    test: GluaNode
    body: list[GluaNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.test.walk()

        for node in self.body:
            yield from node.walk()


@dataclass
class GluaForNumeric(GluaNode):
    var: GluaName
    start: GluaNode
    stop: GluaNode
    step: GluaNode | None
    body: list[GluaNode] = field(default_factory=list)

    def walk(self):
        yield self
        yield from self.var.walk()
        yield from self.start.walk()
        yield from self.stop.walk()

        if self.step is not None:
            yield from self.step.walk()

        for node in self.body:
            yield from node.walk()


@dataclass
class GluaForGeneric(GluaNode):
    targets: list[GluaName]
    iter_expr: GluaNode
    body: list[GluaNode] = field(default_factory=list)

    def walk(self):
        yield self
        for t in self.targets:
            yield from t.walk()
        yield from self.iter_expr.walk()
        for node in self.body:
            yield from node.walk()


@dataclass
class GluaReturn(GluaNode):
    values: list[GluaNode] = field(default_factory=list)

    def walk(self):
        yield self
        for v in self.values:
            yield from v.walk()


@dataclass
class GluaBreak(GluaNode):
    def walk(self):
        yield self


@dataclass
class GluaContinue(GluaNode):
    def walk(self):
        yield self


# endregion


# region Functions
@dataclass
class GluaFuncParam(GluaNode):
    name: str | None
    is_vararg: bool = False

    def walk(self):
        yield self


@dataclass
class GluaFunctionDef(GluaNode):
    name: str | None
    params: list[GluaFuncParam] = field(default_factory=list)
    body: list[GluaNode] = field(default_factory=list)
    is_local: bool = False

    def walk(self):
        yield self
        for p in self.params:
            yield from p.walk()

        for node in self.body:
            yield from node.walk()


# endregion


# region Do-block / Comment
@dataclass
class GluaDoBlock(GluaNode):
    body: list[GluaNode] = field(default_factory=list)

    def walk(self):
        yield self
        for node in self.body:
            yield from node.walk()


@dataclass
class GluaComment(GluaNode):
    text: str

    def walk(self):
        yield self


# endregion

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


@dataclass
class PyIRContext:
    parent_context: "PyIRContext|None" = None
    meta: dict[str, Any] = field(default_factory=dict)
    scope_name: set[str] = field(default_factory=set)


# endregion


# region
@dataclass
class PyIRFile(PyIRNode):
    path: Path | None
    context: PyIRContext
    body: list["PyIRNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for node in self.body:
            yield node


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


# region Function
@dataclass
class PyIRCall(PyIRNode):
    name: str
    args_p: list[PyIRNode] = field(default_factory=list)
    args_kw: dict[str, PyIRNode] = field(default_factory=dict)

    def walk(self):
        yield self
        for arg_p in self.args_p:
            yield arg_p

        for arg_kw in self.args_kw.values():
            yield arg_kw


@dataclass
class PyGmodAPIIRCall(PyIRCall):
    pass


@dataclass
class PyIRFunctionDef(PyIRNode):
    name: str
    signature: dict[str, str | int]
    context: PyIRContext
    body: list["PyIRNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for node in self.body:
            yield node


# endregion


# region Class
@dataclass
class PyIRClassDef(PyIRNode):
    name: str
    context: PyIRContext
    body: list["PyIRNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for node in self.body:
            yield node


# endregion


# region Del
@dataclass
class PyIRDel(PyIRNode):
    value: PyIRNode

    def walk(self):
        yield self
        yield self.value


# endregion


# region Return
@dataclass
class PyIRReturn(PyIRNode):
    value: PyIRNode

    def walk(self):
        yield self
        yield self.value


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


# endreigon

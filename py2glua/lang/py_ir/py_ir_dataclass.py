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
class PyIRFile(PyIRNode):
    path: Path | None
    body: list["PyIRNode"] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def walk(self):
        yield self
        for node in self.body:
            yield node

    def get_vid(self, name: str) -> int:
        n2i = self.meta.setdefault("name_to_id", {})
        i2n = self.meta.setdefault("id_to_name", {})

        if name in n2i:
            return n2i[name]

        vid = len(n2i)
        n2i[name] = vid
        i2n[vid] = name
        return vid


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
    vid: int
    is_global: bool = False

    def walk(self):
        yield self


@dataclass
class PyIRVarUse(PyIRNode):
    vid: int

    def walk(self):
        yield self


# endregion


# region

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

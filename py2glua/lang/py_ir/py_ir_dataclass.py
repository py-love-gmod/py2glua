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

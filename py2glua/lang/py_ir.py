from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class PyIRNode:
    line: int
    offset: int

    def walk(self):
        raise NotImplementedError


# region file
@dataclass
class PyIRFile(PyIRNode):
    body: list[PyIRNode]

    def walk(self):
        yield self
        for node in self.body:
            yield node


# endregion


# region import
class ImportType(Enum):
    EXTERNAL = auto()
    PYTHON_STD = auto()
    INTERNAL = auto()
    UNKNOWN = auto()


@dataclass
class PyIRImport(PyIRNode):
    type: ImportType
    moduel: str


# endregion

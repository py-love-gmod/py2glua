from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class Argument:
    name: str
    type: str = "Any"
    description: str = ""
    kind: Literal["positional", "vararg", "kwarg"] = "positional"
    callback: "FunctionDef | None" = None


@dataclass
class Return:
    type: str = "Any"
    description: str = ""


@dataclass
class FunctionDef:
    name: str
    description: str = ""
    arguments: list[Argument] = field(default_factory=list)
    returns: list[Return] = field(default_factory=list)
    is_static: bool = False
    deprecated: Union[str, bool] | None = None


@dataclass
class ClassDef:
    name: str
    description: str = ""
    parent: str | None = None
    methods: list[FunctionDef] = field(default_factory=list)
    is_static: bool = False
    deprecated: Union[str, bool] | None = None


@dataclass
class EnumItem:
    key: str
    value: str
    description: str = ""


@dataclass
class EnumDef:
    name: str
    description: str = ""
    items: list[EnumItem] = field(default_factory=list)
    deprecated: Union[str, bool] | None = None


@dataclass
class StructDef:
    name: str
    description: str = ""
    fields: list[Argument] = field(default_factory=list)
    deprecated: Union[str, bool] | None = None


SchemaEntry = Union[ClassDef, FunctionDef, EnumDef, StructDef]

from .base_types import Bool, Float, Int, List, Str
from .globals import Global
from .lua_type import LuaType

TYPE_REGISTRY: dict[str, type[LuaType]] = {
    "int": Int,
    "float": Float,
    "str": Str,
    "bool": Bool,
    "list": List,
}

__all__ = [
    "Global",
    "LuaType",
    "TYPE_REGISTRY",
    "Int",
    "Float",
    "Str",
    "Bool",
    "List",
]

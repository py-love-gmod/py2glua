from .base_types import Bool, Float, Int, List, Str
from .lua_type import LuaType

TYPE_REGISTRY: dict[str, type[LuaType]] = {
    "int": Int,
    "float": Float,
    "str": Str,
    "bool": Bool,
    "list": List,
}

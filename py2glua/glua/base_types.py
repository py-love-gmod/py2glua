from .lua_type import LuaType


class Int(LuaType):
    @classmethod
    def lua_check(cls, var: str) -> str:
        return f'assert(type({var}) == "number", "{var} must be int")'


class Float(LuaType):
    @classmethod
    def lua_check(cls, var: str) -> str:
        return f'assert(type({var}) == "number", "{var} must be float")'


class Str(LuaType):
    @classmethod
    def lua_check(cls, var: str) -> str:
        return f'assert(type({var}) == "string", "{var} must be str")'


class Bool(LuaType):
    @classmethod
    def lua_check(cls, var: str) -> str:
        return f'assert(type({var}) == "boolean", "{var} must be bool")'


class List(LuaType):
    @classmethod
    def lua_check(cls, var: str) -> str:
        return f'assert(type({var}) == "table", "{var} must be list")'

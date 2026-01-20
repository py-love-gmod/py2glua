from typing import Any

from .compiler_directive import CompilerDirective


@CompilerDirective.no_compile()
class nil:
    """
    glua-тип отсутствующего значения.
    - `None` означает "пустой объект", то есть значение существует.
    - `nil` в glua означает, что значения нет вообще.

    Переменная, которой присвоен `nil`, считается неопределённой.
    """

    __slots__ = ()

    def __repr__(self):
        return "nil"

    def __str__(self):
        return "nil"

    def __bool__(self):
        return False


@CompilerDirective.no_compile()
class lua_table:
    """
    луа таблица тип.
    В связи с тем, что луа таблица сочетает в себе и свойства словаря, и свойства листа
    для создания подобных сложных объектов придуман этот костыль
    """

    __slots__ = ("array", "map")

    def __init__(
        self,
        array: list[Any] | None = None,
        map: dict[Any, Any] | None = None,
    ) -> None:
        self.array = array or []
        """
        Array-часть Lua-таблицы.
        Интерпретируется как последовательные значения
        с индексами 1..N (Lua-style).
        """

        self.map = map or {}
        """
        Map-часть Lua-таблицы.
        Интерпретируется как явные пары ключ-значение.
        """

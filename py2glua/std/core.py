import enum
from typing import Any, Callable


class RealmType:
    """TODO: Docs"""

    ...


class NilType:
    """TODO: Docs"""

    ...


nil = NilType()
"""TODO: Docs"""


class LuaTable:
    """TODO: Docs"""

    def __init__(self, array_part: list[Any], dict_part: dict[str, Any]) -> None:
        """TODO: Docs"""
        ...

    # TODO:  LuaTable metods


class Realm(enum.Enum):
    """TODO: Docs"""

    SHARED: RealmType
    """TODO: Docs"""

    CLIENT: RealmType
    """TODO: Docs"""

    SERVER: RealmType
    """TODO: Docs"""

    MENU: RealmType
    """TODO: Docs"""


class CompileUtils:
    """'Магический' класс для компилятора.
    Используется в качестве маркеров для особого поведения компиляции"""

    @staticmethod
    def no_compile(func: Callable) -> Callable:
        """Деректива компилятору не компилировать привязанную функцию

        Пример использования:
        ```
            @CompileUtils.no_compile
            def foo(): ...
        ```
        """
        ...

    @staticmethod
    def gmod_api(name: str, realms: list[Realm], *args: Any) -> Callable:
        """Деректива компилятору ПОМЕНЯТЬ имя привязанной функции

        Примеры использования:
        ```
            @CompileUtils.gmod_api("IsValid", [Realm.SHARED])
            def foo(): ...

            foo()
        ```
            ->
        ```
            IsValid()
        ```

        ---
        ```
            @CompileUtils.gmod_api("IsValid", [Realm.SHARED], 1)
            def foo(number): ...

            foo(2)
        ```
            ->
        ```
            IsValid(1, 2)
        ```

        ---
        Args:
            name str: Имя в которое будет транслировано
            realms list[Realm]: Реалмы в которых данный метод работает
            *args Any: Параметры которые будут подставлены по дефолту

        """
        ...

    @staticmethod
    def complex_enum(mapping: list[tuple[str, Any, bool]]) -> Callable:
        """Используется для создания более комлпекстных типов ENUM из классов

        Пример использования:
        ```
            @CompileUtils.complex_enum([
                ("a", 1, False),
                ("b", "boo", False),
                ("c", "_G", True),
                ("d", [1,2,3], False)
            ])
            class CustomEnum:
                a: str
                b: str
                c: object
                d: list

            CustomEnum.a
            CustomEnum.b
            CustomEnum.c
            CustomEnum.d
        ```
            ->
        ```
            1
            "foo"
            _G
            {1,2,3}
        ```

        Args:
            mapping list[tuple[str, Any, bool]]: Соотношение поля и значения, где:
                [0] - Имя поля
                [1] - Значение
                [2] - Является ли глобальным (работает только со строками)
        """
        ...

    DEBUG_BUILD: bool
    """Флаг сборки для `debug` режима.
    В отличии от рантайм `developer` переменной является литералом определённым на момент запуска компиляции
    
    `py2glua build -d` => `CompileUtils.DEBUG_BUILD == True`
    
    `py2glua build` => `CompileUtils.DEBUG_BUILD == False`
    """

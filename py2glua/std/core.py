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

    def __init__(
        self,
        array_part: list[Any],
        dict_part: dict[str, Any],
    ) -> None:
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
    def gmod_api(name: str, realms: list[RealmType], *args: Any) -> Callable:
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
    def complex_enum(
        mapping: list[tuple[str, Any, bool] | tuple[str, Any]],
    ) -> Callable:
        """Используется для создания более комлпекстных типов ENUM из классов

        Пример использования:
        ```
            @CompileUtils.complex_enum([
                ("a", 1, False),
                ("b", "boo", False),
                ("c", "_G", True),
                ("d", [1,2,3])
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
            mapping list[tuple[str, Any, bool] | tuple[str, Any]]: Соотношение поля и значения, где:
                [0] - Имя поля
                [1] - Значение
                [2] - Является ли глобальным (опционально, работает только со строками, по умолчанию `False`)
        """
        ...

    @staticmethod
    def anonymous(func: Callable) -> Callable:
        """Превращает фунцию в анонимную

        Пример использования:
        ```
            @CompileUtils.anonymous
            def foo(): ...

            foo()
        ```
            ->
        ```
            function() ... end
        ```

        """
        ...

    @staticmethod
    def with_as_if(func: Callable) -> Callable:
        """Позволяет использовать функцию или класс как if через with

        Пример использования:
        ```
            @CompileUtils.with_as_if
            def foo(): ...

            with foo(): ...
        ```
            ->
        ```
            if foo() ... end
        ```
        """

        ...

    @staticmethod
    def with_macro(func: Callable) -> Callable:
        """Позволяет пометить функцию как макрос развёртки

        Пример использования:
        ```
            @CompileUtils.with_macro
            def foo():
                pre
                CompileUtils.WITH_MACRO_BODY
                post

            with foo():
                body
        ```
            ->
        ```
            pre
            body
            pos
        ```
        """
        ...

    WITH_MACRO_BODY: object
    """Маркер использующийся в :meth:`with_macro`"""

    DEBUG_BUILD: bool
    """Флаг сборки для `debug` режима.
    В отличии от рантайм `developer` переменной является литералом определённым на момент запуска компиляции
    
    `py2glua build -d` => `CompileUtils.DEBUG_BUILD == True`
    
    `py2glua build` => `CompileUtils.DEBUG_BUILD == False`
    """

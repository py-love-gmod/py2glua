from collections.abc import Callable
from typing import Literal


class CompilerDirective:
    """Класс отвечающий за дерективы компилятору"""

    # region debug
    DEBUG: bool
    """Переменная отвечающая за дебаг сборку
    
    - True - debug
    - False - release
    """

    @staticmethod
    def debug_compile_only() -> Callable:
        """Декоратор отвечающий за использование данной функции только в debug режиме"""

        def decorator(fn):
            return fn

        return decorator

    # endregion

    class RealmMarker:
        """Маркер обозначающий что данная переменная отвечает за реалм"""

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc, tb):
            pass

    # region compile func type
    @staticmethod
    def lazy_compile() -> Callable:
        """Помечает метод или класс как "ленивый" для компилятора

        Код, помеченный как lazy, будет включён в итоговый glua только если
        его использование может быть обнаружено статическим анализом IR
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def inline() -> Callable:
        """Просьба компилятору вставлять тело функции на место вызова (inline)"""

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def no_compile() -> Callable:
        """Помечает метод или класс как некомпилируемый

        Данный декоратор исключает питон реализацию из вывода glua

        Внутренний декоратор. Используется py2glua для исключения кода из вывода
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def anonymous() -> Callable:
        """Указывает что функция является анонимной, и её тело необходимо поставить ака function(args) body end"""

        def decorator(fn):
            return fn

        return decorator

    # endregion

    @staticmethod
    def gmod_api(name: str, realm: list[RealmMarker], method: bool = False) -> Callable:
        """Помечает функцию или класс как элемент GMod API

        Делает две вещи:
        - присваивает указанное имя в таблице API
        - исключает Python-реализацию из компиляции (аналогично no_compile)

        Внутренний инструмент для генерации API-обёрток
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def gmod_prototype(fold: str) -> Callable:
        """TODO:

        Обязазательно реализовывать поле `uid`
        
        Обязазательно реализовывать метод `overrid(realm, fn) -> callable`
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def typeguard_nil() -> Callable:
        """Помечает функцию как валидатор Nil. То есть после этой функции метод точно не Nil. Сделано для IsValid"""

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def gmod_special_enum(
        fields: dict[
            str,
            tuple[
                Literal["str", "bool", "int", "global"],
                str | bool | int,
            ],
        ],
    ) -> Callable:
        """Делает магию с енумами нестандартными.
        Путь указания маппинга
        "Поле": ["тип поля", значение]

        Для примера использования смотрите glua.realm
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def with_condition() -> Callable:
        """
        Помечает класс как источник условных with-блоков.

        Конструкция:
            with Class.attr:
                body

        Интерпретируется компилятором как:
            if Class.attr:
                body

        """

        def decorator(fn):
            return fn

        return decorator

    # region contextmanager
    @staticmethod
    def contextmanager() -> Callable:
        """Указывает что данная функция обязана реализовывать конструкцию with"""

        def decorator(fn):
            return fn

        return decorator

    contextmanager_body = object()
    """Указание компилятору что в данном месте для конструкции with необходимо подставить само тело блока"""
    # endregion

    @staticmethod
    def raw(lua_code: str) -> None:
        """Позволяет сказать компилятору "вставь голый луа сюдa". Что либо возвращать из этого блока нельзя"""
        pass

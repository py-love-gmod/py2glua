from collections.abc import Callable
from typing import Literal


class CompilerDirective:
    """Класс отвечающий за дерективы компилятору"""

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


class InternalCompilerDirective:
    """Внутренние декораторы для py2glua"""

    @staticmethod
    def stub() -> Callable:
        """
        Помечает метод как заглушку.
        Вся внутренняя реализация будет убрана во время pass
        Так же будут удалены и комментарии
        """

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
    def gmod_api(name: str, realm: list["RealmMarker"]) -> Callable:
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
    def contextmanager() -> Callable:
        """Указывает что данная функция обязана реализовывать конструкцию with"""

        def decorator(fn):
            return fn

        return decorator

    contextmanager_body = object()
    """Указание компилятору что в данном месте для конструкции with необходимо подставить само тело блока"""

    @staticmethod
    def std_lib_obj() -> Callable:
        """Помечает функцию или класс как std_lib объект

        Запрещено использование вне py2glua
        """

        def decorator(fn):
            return fn

        return decorator


@InternalCompilerDirective.no_compile()
class RealmMarker:
    pass

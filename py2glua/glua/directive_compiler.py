from collections.abc import Callable


class CompilerDirective:
    """Класс отвечающий за дерективы компилятору"""

    DEBUG: bool
    """Переменная отвечающая за дебаг сборку. True - debug, False - release"""

    @staticmethod
    def debug_compile_only() -> Callable:
        """Декоратор отвечающий за использование данной функции только в debug режиме"""

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def lazy_compile() -> Callable:
        """Помечает метод или класс как "ленивый" для компилятора.

        Код, помеченный как lazy, будет включён в итоговый glua только если
        его использование может быть обнаружено статическим анализом IR.
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
    def no_compile() -> Callable:
        """Помечает метод или класс как некомпилируемый.

        Данный декоратор исключает питон реализацию из вывода glua

        Внутренний декоратор. Используется py2glua для исключения кода из вывода
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def gmod_api(name: str) -> Callable:
        """Помечает функцию или класс как элемент GMod API.

        Делает две вещи:
        - присваивает указанное имя в таблице API
        - исключает Python-реализацию из компиляции (аналогично no_compile)

        Внутренний инструмент для генерации API-обёрток
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def enum_name_gmod() -> Callable:
        """Помечает класс Enum как "используй имена как константы глуа".

        По факту создан только для Reaml enum класса.

        Внутренний инструмент для кодгена

        p.s. Я не хочу хардкодить чисто realm в вывод по имени. Слишком много чести ему
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def std_lib_obj() -> Callable:
        """Помечает функцию или класс как std_lib объект

        Запрещено использование вне py2glua
        """

        def decorator(fn):
            return fn

        return decorator

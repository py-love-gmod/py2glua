from collections.abc import Callable


class InternalCompilerDirective:
    """Внутренние декораторы для py2glua

    Запрещено использование вне py2glua
    """

    @staticmethod
    def std_lib_obj() -> Callable:
        """Помечает функцию или класс как std_lib объект"""

        def decorator(fn):
            return fn

        return decorator

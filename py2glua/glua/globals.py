from collections.abc import Callable
from typing import Any

from .directive_compiler import InternalCompilerDirective


@InternalCompilerDirective.stub()
class Global:
    @staticmethod
    def var(value: Any, external: bool = False) -> Any:
        """Помечает переменную как глобальную для компилятора.
        В случае если не был установлен флаг external, и переменная не используется внутри проекта - ложит билд.

        Args:
            value (Any): Любое значение которое будет присвоено переменной
            external (bool, optional): Необходимо ли эта переменная для внешнего апи аддона? Defaults to False.
        """
        pass

    @staticmethod
    def mark(external: bool = False) -> Callable:
        """Помечает класс/функцию/метод как глобальную для компилятора.
        В случае если не был установлен флаг external, и переменная не используется внутри проекта - ложит билд.

        Args:
            external (bool, optional): Необходимо ли этот класс/функция/метод для внешнего апи аддона? Defaults to False.
        """

        def decorator(fn):
            return fn

        return decorator

from collections.abc import Callable
from typing import Any

from .compiler_directive import CompilerDirective


@CompilerDirective.no_compile()
class Global:
    @staticmethod
    def var() -> Any:
        """Помечает переменную как внешне глобально доступную."""
        pass

    @staticmethod
    def mark() -> Callable:
        """Помечает класс/функцию/метод как внешне глобально доступную."""

        def decorator(fn):
            return fn

        return decorator

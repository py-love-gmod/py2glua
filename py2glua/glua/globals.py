from collections.abc import Callable
from typing import Any


class Global:
    @staticmethod
    def var(value: Any) -> Any:
        """Помечает переменную как глобальную. Используется только при создании новой переменной"""
        return value

    @staticmethod
    def mark() -> Callable:
        """Помечает класс или функцию как глобальную"""

        def decorator(class_cls):
            return class_cls

        return decorator

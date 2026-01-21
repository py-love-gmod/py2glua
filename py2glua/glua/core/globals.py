from collections.abc import Callable
from typing import Any

from .compiler_directive import CompilerDirective


@CompilerDirective.no_compile()
class Global:
    @staticmethod
    def create_external_var() -> Any:
        """
        Компиляторная директива.

        Помечает переменную как внешне доступную для использования за пределами аддона.

        НЕ делает символ глобальным во время выполнения.
        """
        pass

    @staticmethod
    def create_external_def() -> Callable:
        """
        Компиляторный декоратор.

        Помечает функцию, метод как внешне доступный символ.

        НЕ делает символ глобальным во время выполнения.
        """

        def decorator(fn):
            return fn

        return decorator

    @staticmethod
    def use(ref: Any) -> Any:
        """
        Компиляторная ссылка на символ.

        Явно указывает компилятору, что данный символ
        должен быть найден и использован вне текущего скоупа.
        """
        pass

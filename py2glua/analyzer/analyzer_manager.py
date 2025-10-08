from ast import AST
from collections.abc import Callable
from typing import Type

from .base_analyzer_unit import BaseAnalyzerUnit


class AnalyzerManager:
    """Класс для управления всеми анализаторами"""

    _analyzer_units: dict[Type[AST], BaseAnalyzerUnit] = {}

    @classmethod
    def register(cls, node_type: Type[AST]) -> Callable:
        """Декоратор для регистрация класса для конвертора

        Args:
            node_type (Type[ast.AST]): Тип АСТ ноды

        Returns:
            Callable: декоратор
        """

        def decorator(convertor_cls: Type[BaseAnalyzerUnit]):
            cls._analyzer_units[node_type] = convertor_cls()
            return convertor_cls

        return decorator

    @classmethod
    def get(cls, node: AST) -> BaseAnalyzerUnit | None:
        """Возвращает зарегестрованный класс

        Args:
            node (AST): АСТ нода

        Returns:
            BaseAnalyzerUnit | None: Класс или None
        """
        return cls._analyzer_units.get(type(node), None)

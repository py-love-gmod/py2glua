from ast import AST
from collections.abc import Callable
from typing import Type

from .base_convertor import BaseConvertor


class ConvertorManager:
    """Класс для управления всеми конверторами"""

    _convertors: dict[Type[AST], BaseConvertor] = {}

    @classmethod
    def register(cls, node_type: Type[AST]) -> Callable:
        """Декоратор для регистрация класса для конвертора

        Args:
            node_type (Type[ast.AST]): Тип АСТ ноды

        Returns:
            Callable: декоратор
        """

        def decorator(convertor_cls: Type[BaseConvertor]):
            cls._convertors[node_type] = convertor_cls()
            return convertor_cls

        return decorator

    @classmethod
    def get(cls, node: AST) -> BaseConvertor | None:
        """Возвращает зарегестрованный класс

        Args:
            node (AST): АСТ нода

        Returns:
            BaseConvertor | None: Класс или None
        """
        return cls._convertors.get(type(node), None)

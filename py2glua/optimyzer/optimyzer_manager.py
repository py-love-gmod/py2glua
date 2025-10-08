from ast import AST
from collections.abc import Callable
from typing import Type

from .base_optimyzer_unit import BaseOptimyzerUnit
from .optimization_level import OptimizationLevel


class OptimyzerManager:
    """Класс для управления всеми оптимизаторами"""

    _optimzer_units: dict[OptimizationLevel, dict[Type[AST], BaseOptimyzerUnit]] = {}

    @classmethod
    def register(
        cls,
        node_type: Type[AST],
        optimization_level: OptimizationLevel,
    ) -> Callable:
        """Декоратор для регистрации юнита оптимизации

        Args:
            node_type (Type[ast.AST]): Тип ноды AST
            optimization_level (OptimizationLevel): Уровень оптимизации

        Returns:
            Callable: Декоратор
        """

        def decorator(convertor_cls: Type[BaseOptimyzerUnit]):
            cls._optimzer_units.setdefault(optimization_level, {})
            cls._optimzer_units[optimization_level][node_type] = convertor_cls()
            return convertor_cls

        return decorator

    @classmethod
    def get(
        cls,
        optimization_level: OptimizationLevel,
        node: AST,
    ) -> BaseOptimyzerUnit | None:
        """Возвращает зарегестрованный класс

        Args:
            optimization_level (OptimizationLevel): Необходимый уровень оптимизации
            node (AST): АСТ нода

        Returns:
            BaseOptimyzerUnit | None: Класс или None
        """
        opt_lvl = cls._optimzer_units.get(optimization_level, None)
        if opt_lvl:
            return opt_lvl.get(type(node), None)

        return None

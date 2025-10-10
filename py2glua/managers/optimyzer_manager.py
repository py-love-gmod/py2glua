from ast import AST
from collections.abc import Callable
from typing import Any, Generic, Type, TypeVar

from ..optimyzer import OptimizationLevel
from .registry_manager import RegistryManager

T = TypeVar("T")


class BaseOptimyzerUnit(Generic[T]):
    def process(self, node: T, ctx: dict) -> Any:
        raise NotImplementedError


class OptimyzerManager(RegistryManager["BaseOptimyzerUnit"]):
    _units: dict[OptimizationLevel, dict[Type[AST], T]] = {}  # type: ignore

    @classmethod
    def register(
        cls,
        opt_level: OptimizationLevel,
        node_type: Type[AST],
        *args,
        **kwargs,
    ) -> Callable[[Type[T]], Type[T]]:
        def decorator(unit_cls: Type[T]):
            cls._units[opt_level][node_type] = unit_cls()
            return unit_cls

        return decorator

    @classmethod
    def get(cls, opt_level: OptimizationLevel, node: AST, *args, **kwargs) -> Any:
        opt_region = cls._units.get(opt_level)
        if opt_region is None:
            return None

        return opt_region.get(type(node), None)

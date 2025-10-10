from typing import Generic, TypeVar

from .registry_manager import RegistryManager

T = TypeVar("T")


class BaseOptimyzerUnit(Generic[T]):
    def process(self, node: T, ctx: dict) -> str:
        raise NotImplementedError


class OptimyzerManager(RegistryManager["BaseOptimyzerUnit"]):
    pass

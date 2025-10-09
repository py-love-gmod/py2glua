from typing import Generic, TypeVar

from .registry_manager import RegistryManager

T = TypeVar("T")


class BaseAnalyzerUnit(Generic[T]):
    def process(self, node: T, ctx: dict) -> None:
        raise NotImplementedError


class AnalyzerManager(RegistryManager["BaseAnalyzerUnit"]):
    pass

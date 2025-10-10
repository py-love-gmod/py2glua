from typing import Generic, TypeVar

from .registry_manager import RegistryManager

T = TypeVar("T")


class BaseConvertorUnit(Generic[T]):
    def process(self, node: T, ctx: dict, indent: int = 0) -> str:
        raise NotImplementedError


class ConvertorManager(RegistryManager["BaseConvertorUnit"]):
    pass

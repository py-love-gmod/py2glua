from ast import AST
from collections.abc import Callable
from typing import Any, Generic, Type, TypeVar

T = TypeVar("T")


class RegistryManager(Generic[T]):
    _units: dict[Type[AST], T] = {}

    @classmethod
    def register(
        cls,
        node_type: Type[AST],
        *args,
        **kwargs,
    ) -> Callable[[Type[T]], Type[T]]:
        def decorator(unit_cls: Type[T]):
            cls._units[node_type] = unit_cls()
            return unit_cls

        return decorator

    @classmethod
    def get(cls, node: AST, *args, **kwargs) -> Any:
        return cls._units.get(type(node))

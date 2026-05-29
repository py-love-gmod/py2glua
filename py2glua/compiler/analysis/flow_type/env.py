from __future__ import annotations

from typing import Dict, Optional

from .types import AnyType, Type


class TypeEnv:
    def __init__(self, parent: Optional[TypeEnv] = None):
        self.parent = parent
        self._vars: Dict[str, Type] = {}

    def set(self, name: str, typ: Type):
        self._vars[name] = typ

    def get(self, name: str) -> Type:
        if name in self._vars:
            return self._vars[name]

        if self.parent:
            return self.parent.get(name)

        return AnyType

    def push(self) -> TypeEnv:
        return TypeEnv(parent=self)

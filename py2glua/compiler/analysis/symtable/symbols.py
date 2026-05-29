from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from plg_reader import IRNode


@dataclass
class IRSymbolRef(IRNode):
    name: str
    symbol_id: int
    kind: str  # 'function', 'class', 'variable', 'param', 'import', 'module'


@dataclass
class Symbol:
    id: int
    name: str
    kind: str  # 'function', 'class', 'variable', 'param', 'import', 'module'
    definition: Optional[IRNode] = None
    uses: list[IRSymbolRef] = field(default_factory=list)
    read_count: int = 0
    write_count: int = 0


class SymbolTable:
    def __init__(self) -> None:
        self._symbols: dict[int, Symbol] = {}

    def create(self, name: str, kind: str, definition: IRNode | None = None) -> Symbol:
        new_id = len(self._symbols) + 1
        sym = Symbol(id=new_id, name=name, kind=kind, definition=definition)
        self._symbols[new_id] = sym
        return sym

    def get(self, symbol_id: int) -> Symbol:
        return self._symbols[symbol_id]

    def all(self) -> list[Symbol]:
        return list(self._symbols.values())


class Scope:
    def __init__(self, parent: Scope | None = None, name: str = "") -> None:
        self.parent = parent
        self.symbols: dict[str, tuple[int, str]] = {}
        self.name = name

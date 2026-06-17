from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from plg_reader import IRFile, IRNode

ScopeKind = Literal["module", "class", "function", "builtins"]
SymbolKind = Literal[
    "module", "class", "function", "variable", "param", "global", "imported"
]


@dataclass
class SymbolRefIR(IRNode):
    symbol: Symbol


@dataclass
class Scope:
    parent: Scope | None
    symbols: dict[str, Symbol]
    kind: ScopeKind
    module_name: str = ""
    self_symbol: Symbol | None = None


@dataclass
class Symbol:
    name: str
    scope: Scope
    kind: SymbolKind
    node: IRNode
    qualified_name: str = ""
    references: list[IRNode] = field(default_factory=list)
    scope_ref: Scope | None = None


@dataclass
class CompilationUnit:
    irs: dict[str, IRFile]
    module_scopes: dict[str, Scope]
    node_to_scope: dict[int, Scope]
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)

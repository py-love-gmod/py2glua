from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ....py.ir_dataclass import PyIRFile, PyIRNode

if TYPE_CHECKING:
    from .collect_simbols import CollectedSymbol, FileSymbolTable

EnumValueKind = Literal["int", "str", "bool", "global"]


@dataclass(frozen=True)
class EnumValue:
    name: str
    kind: EnumValueKind
    value: str | int | bool


@dataclass
class EnumInfo:
    class_symbol_id: int
    fqname: str
    values: dict[str, EnumValue]
    origin: Literal["python_enum", "gmod_special_enum"]


@dataclass(frozen=True)
class SymbolInfo:
    id: int
    fqname: str
    symbol: CollectedSymbol
    file: Path

    def __repr__(self) -> str:
        return f"<SymbolInfo {self.id:03d} {self.fqname} ({self.file})>"


@dataclass
class InlineRecursionState:
    done: bool = False
    processed_files: set[Path] = field(default_factory=set)
    inline_funcs: set[int] = field(default_factory=set)
    raw_edges: dict[int, set[int]] = field(default_factory=dict)
    locations: dict[int, tuple[Path, int | None, int | None]] = field(
        default_factory=dict
    )


@dataclass
class ContextManagerInfo:
    function_symbol_id: int
    fqname: str
    file: Path
    line: int | None
    offset: int | None
    pre_body: list[PyIRNode]
    post_body: list[PyIRNode]


@dataclass
class DeprecatedSymbolInfo:
    symbol_id: int
    fqname: str
    message: str | None
    file: Path | None
    line: int | None
    offset: int | None


class AnalysisContext:
    def __init__(self) -> None:
        self.file_simbol_data: dict[Path, FileSymbolTable] = {}

        self.symbols: dict[int, SymbolInfo] = {}
        self.symbol_id_by_fqname: dict[str, int] = {}
        self.symbol_ids_by_name: dict[str, list[int]] = {}

        self._next_symbol_id: int = 1

        self.enums: dict[int, EnumInfo] = {}

        self.inline_recursion_state: InlineRecursionState = InlineRecursionState()

        self.with_condition_classes: set[int] = set()

        self.context_managers: dict[int, ContextManagerInfo] = {}

        self.deprecated_symbols: dict[int, DeprecatedSymbolInfo] = {}
        self.deprecated_members: dict[tuple[int, str], DeprecatedSymbolInfo] = {}
        self.deprecated_warned: set[tuple[Path | None, int | None, int | None, str]] = (
            set()
        )

    def new_symbol_id(self) -> int:
        sid = self._next_symbol_id
        self._next_symbol_id += 1
        return sid


class AnalysisPass:
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        raise NotImplementedError

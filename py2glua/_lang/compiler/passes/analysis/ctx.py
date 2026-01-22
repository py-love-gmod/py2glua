from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import PyIRNode

if TYPE_CHECKING:
    from .collect_simbols import CollectedSymbol, FileSymbolTable


@dataclass(frozen=True)
class SymbolInfo:
    id: int
    fqname: str
    symbol: CollectedSymbol
    file: Path

    def __repr__(self) -> str:
        return f"<SymbolInfo {self.id:03d} {self.fqname} ({self.file})>"


class AnalysisContext:
    def __init__(self) -> None:
        self.file_simbol_data: dict[Path, FileSymbolTable] = {}

        self.symbols: dict[int, SymbolInfo] = {}
        self.symbol_id_by_fqname: dict[str, int] = {}
        self.symbol_ids_by_name: dict[str, list[int]] = {}

        self._next_symbol_id: int = 1

        self.enum_map: dict[int, dict[str, PyIRNode]] = {}
        self.with_map: dict[int, Any] = {}
        self.api_map: dict[int, Any] = {}

    def new_symbol_id(self) -> int:
        sid = self._next_symbol_id
        self._next_symbol_id += 1
        return sid


class AnalysisPass:
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        raise NotImplementedError

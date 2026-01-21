from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ....py.ir_builder import PyIRFile

if TYPE_CHECKING:
    from .collect_simbols import FileSymbol, FileSymbolTable


@dataclass(frozen=True)
class SymbolInfo:
    id: int
    fqname: str
    symbol: FileSymbol
    file: Path


class AnalysisContext:
    def __init__(self):
        self.file_simbol_data: dict[Path, FileSymbolTable] = {}

        self.symbols: dict[int, SymbolInfo] = {}
        self.symbol_id_by_fqname: dict[str, int] = {}
        self.symbol_ids_by_name: dict[str, list[int]] = {}

        self._next_symbol_id: int = 1


class AnalysisPass:
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        raise NotImplementedError

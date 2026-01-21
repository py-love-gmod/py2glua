from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRClassDef,
    PyIRFunctionDef,
    PyIRVarUse,
)
from .ctx import AnalysisContext, AnalysisPass


# Model
@dataclass(frozen=True)
class FileSymbol:
    name: str
    kind: Literal["function", "class", "var"]
    signature: str | None
    decorators: tuple[str, ...]

    def __repr__(self) -> str:
        parts: list[str] = [self.kind, self.name]

        if self.signature is not None:
            parts.append(self.signature)

        if self.decorators:
            parts.append("@" + ",".join(self.decorators))

        return "<Symbol " + " ".join(parts) + ">"


@dataclass(frozen=True)
class FileSymbolTable:
    file: Path
    symbols: tuple[FileSymbol, ...]

    def __repr__(self) -> str:
        lines: list[str] = [f"<FileSymbolTable {self.file}>"]

        if not self.symbols:
            lines.append("  <no public symbols>")
            return "\n".join(lines)

        for s in self.symbols:
            lines.append(f"  {repr(s)}")

        return "\n".join(lines)


# Collector
class FileSymbolCollector:
    @staticmethod
    def _format_function_signature(
        sig: dict[str, tuple[str | None, str | None]],
    ) -> str:
        parts: list[str] = []

        for name, (ann, default) in sig.items():
            item = name

            if ann is not None:
                item += f": {ann}"

            if default is not None:
                item += f" = {default}"

            parts.append(item)

        return f"({', '.join(parts)})"

    @staticmethod
    def collect(ir: PyIRFile) -> FileSymbolTable:
        assert ir.path is not None

        symbols: list[FileSymbol] = []

        for node in ir.body:
            if isinstance(node, PyIRFunctionDef):
                if node.name.startswith("_"):
                    continue

                symbols.append(
                    FileSymbol(
                        name=node.name,
                        kind="function",
                        signature=FileSymbolCollector._format_function_signature(
                            node.signature
                        ),
                        decorators=tuple(d.name for d in node.decorators),
                    )
                )
                continue

            if isinstance(node, PyIRClassDef):
                if node.name.startswith("_"):
                    continue

                symbols.append(
                    FileSymbol(
                        name=node.name,
                        kind="class",
                        signature="()",
                        decorators=tuple(d.name for d in node.decorators),
                    )
                )
                continue

            if isinstance(node, PyIRAssign):
                for tgt in node.targets:
                    if isinstance(tgt, PyIRVarUse) and not tgt.name.startswith("_"):
                        symbols.append(
                            FileSymbol(
                                name=tgt.name,
                                kind="var",
                                signature=None,
                                decorators=(),
                            )
                        )

        return FileSymbolTable(
            file=ir.path,
            symbols=tuple(symbols),
        )


# Analysis pass
class CollectSymbolsPass(AnalysisPass):
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        table = FileSymbolCollector.collect(ir)
        ctx.file_simbol_data[table.file] = table

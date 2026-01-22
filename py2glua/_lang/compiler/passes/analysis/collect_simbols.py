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

# Models
SymbolKind = Literal["function", "class", "var", "arg"]
SymbolScope = Literal["module", "function", "class"]


@dataclass(frozen=True)
class CollectedSymbol:
    name: str
    kind: SymbolKind
    scope: SymbolScope
    owner: str | None
    signature: str | None
    decorators: tuple[str, ...]

    def __repr__(self) -> str:
        base = f"{self.scope}:{self.kind} {self.name}"
        if self.owner:
            base += f" @{self.owner}"
        if self.signature:
            base += f" {self.signature}"
        if self.decorators:
            base += " @" + ",".join(self.decorators)
        return f"<Symbol {base}>"


@dataclass(frozen=True)
class FileSymbolTable:
    file: Path
    symbols: tuple[CollectedSymbol, ...]

    def __repr__(self) -> str:
        lines: list[str] = [f"<FileSymbolTable {self.file}>"]
        if not self.symbols:
            lines.append("  <no symbols>")
            return "\n".join(lines)

        for sym in self.symbols:
            lines.append(f"  {sym}")
        return "\n".join(lines)


# Collector
class FileSymbolCollector:
    RETURN_KEY = "__return__"

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

    @classmethod
    def collect(cls, ir: PyIRFile) -> FileSymbolTable:
        assert ir.path is not None

        symbols: list[CollectedSymbol] = []

        def walk_block(
            body,
            scope: SymbolScope,
            owner: str | None,
        ) -> None:
            for node in body:
                # Function def
                if isinstance(node, PyIRFunctionDef):
                    symbols.append(
                        CollectedSymbol(
                            name=node.name,
                            kind="function",
                            scope=scope,
                            owner=owner,
                            signature=cls._format_function_signature(node.signature),
                            decorators=tuple(d.name for d in node.decorators),
                        )
                    )

                    # arguments
                    for arg in node.signature.keys():
                        symbols.append(
                            CollectedSymbol(
                                name=arg,
                                kind="arg",
                                scope="function",
                                owner=node.name,
                                signature=None,
                                decorators=(),
                            )
                        )

                    walk_block(
                        node.body,
                        scope="function",
                        owner=node.name,
                    )
                    continue

                # Class def
                if isinstance(node, PyIRClassDef):
                    symbols.append(
                        CollectedSymbol(
                            name=node.name,
                            kind="class",
                            scope=scope,
                            owner=owner,
                            signature="()",
                            decorators=tuple(d.name for d in node.decorators),
                        )
                    )

                    walk_block(
                        node.body,
                        scope="class",
                        owner=node.name,
                    )
                    continue

                # Assignment
                if isinstance(node, PyIRAssign):
                    for tgt in node.targets:
                        if isinstance(tgt, PyIRVarUse):
                            symbols.append(
                                CollectedSymbol(
                                    name=tgt.name,
                                    kind="var",
                                    scope=scope,
                                    owner=owner,
                                    signature=None,
                                    decorators=(),
                                )
                            )

        walk_block(
            ir.body,
            scope="module",
            owner=None,
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

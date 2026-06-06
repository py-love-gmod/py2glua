from __future__ import annotations

from typing import Callable, Type

from plg_reader import IRFile, IRNode

from ..analysis import SymbolTable

_EMITTERS: dict[Type[IRNode], Callable] = {}
_INDENT_STR = "    "


def emit(node: IRNode, lines: list[str], indent: int) -> None:
    emitter = _EMITTERS.get(type(node))
    if emitter is not None:
        emitter(node, lines, indent)

    else:
        prefix = _INDENT_STR * indent
        lines.append(f"{prefix}-- [[ {type(node).__name__} ]] {node}")


def dump_to_lua(ir_file: IRFile, symbol_table: SymbolTable | None = None) -> str:
    lines = []
    if symbol_table is not None:
        lines.append("-- Symbols:")
        for sym in symbol_table.all():
            lines.append(f"--   {sym.id}: {sym.name} ({sym.kind})")

        lines.append("")

    for node in ir_file.body:
        emit(node, lines, 0)

    if ir_file.imports:
        import_lines = []
        for imp in ir_file.imports:
            if imp.is_from:
                names = ", ".join(
                    item[1] if isinstance(item, tuple) else item for item in imp.names
                )
                import_lines.append(f"-- from {'.'.join(imp.modules)} import {names}")

            else:
                for mod, name in zip(imp.modules, imp.names):
                    alias = name[1] if isinstance(name, tuple) else name
                    import_lines.append(f"-- import {mod} as {alias}")

        lines = import_lines + [""] + lines

    return "\n".join(lines)

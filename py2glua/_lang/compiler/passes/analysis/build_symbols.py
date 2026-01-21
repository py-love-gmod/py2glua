from __future__ import annotations

from pathlib import Path

from ....py.ir_builder import PyIRFile
from .ctx import AnalysisContext, AnalysisPass, SymbolInfo


class BuildSymbolIndexPass(AnalysisPass):
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if ctx.symbol_id_by_fqname:
            return

        for table in ctx.file_simbol_data.values():
            module = cls._module_name_from_path(table.file)

            for sym in table.symbols:
                fq = cls._make_fqname(module, table.file, sym.name)
                sid = ctx.symbol_id_by_fqname.get(fq)
                if sid is None:
                    sid = ctx._next_symbol_id
                    ctx._next_symbol_id += 1

                    ctx.symbol_id_by_fqname[fq] = sid
                    ctx.symbol_ids_by_name.setdefault(sym.name, []).append(sid)
                    ctx.symbols[sid] = SymbolInfo(
                        id=sid,
                        fqname=fq,
                        symbol=sym,
                        file=table.file,
                    )

    @staticmethod
    def _make_fqname(module: str | None, file: Path, name: str) -> str:
        if module is None:
            return f"<file:{file.as_posix()}>.{name}"

        return f"{module}.{name}"

    @staticmethod
    def _module_name_from_path(path: Path) -> str | None:
        parts = path.parts

        try:
            i = len(parts) - 1 - list(reversed(parts)).index("py2glua")

        except ValueError:
            return None

        tail = list(parts[i:])
        if not tail:
            return None

        last = tail[-1]
        if last.endswith(".py"):
            name = last[:-3]
            if name == "__init__":
                tail = tail[:-1]
                
            else:
                tail[-1] = name

        if not tail:
            return None

        return ".".join(tail)

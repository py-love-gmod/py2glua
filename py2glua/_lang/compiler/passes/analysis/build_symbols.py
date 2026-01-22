from __future__ import annotations

from pathlib import Path

from .....config import Py2GluaConfig
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
                if sym.scope != "module":
                    continue

                if sym.kind in ("arg",):
                    continue

                fq = cls._make_fqname(module, table.file, sym.name)

                if fq in ctx.symbol_id_by_fqname:
                    continue

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

    @classmethod
    def _module_name_from_path(cls, path: Path) -> str | None:
        p = path.resolve()

        parts = p.parts
        last_py2glua = cls._rindex(parts, "py2glua")
        if last_py2glua is not None:
            tail = list(parts[last_py2glua:])
            return cls._normalize_tail(tail)

        try:
            src_root = Py2GluaConfig.source
            rel = p.relative_to(src_root)

        except Exception:
            return None

        return cls._normalize_tail(list(rel.parts))

    @staticmethod
    def _rindex(parts: tuple[str, ...], value: str) -> int | None:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == value:
                return i

        return None

    @staticmethod
    def _normalize_tail(tail: list[str]) -> str | None:
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

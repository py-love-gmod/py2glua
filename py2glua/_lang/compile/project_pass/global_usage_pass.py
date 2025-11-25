from __future__ import annotations

from ...py.py_ir_dataclass import (
    PyIRFile,
    PyIRVarUse,
)
from ..compile_exception import CompileError
from ..compiler import Compiler


class GlobalUsagePass:
    @classmethod
    def run(cls, project: list[PyIRFile], compiler: Compiler) -> list[PyIRFile]:
        declared: dict[str, list[tuple[PyIRFile, bool]]] = {}
        used: set[str] = set()

        for f in project:
            globs = f.context.meta.get("globals", {})
            for name, info in globs.items():
                declared.setdefault(name, []).append((f, info.get("external", False)))

        for f in project:
            for node in f.walk():
                if isinstance(node, PyIRVarUse):
                    used.add(node.name)

        for name, occ_list in declared.items():
            if name in used:
                continue

            for f, external in occ_list:
                if not external:
                    raise CompileError(
                        f"Глобальная переменная/функция '{name}' объявлена, "
                        f"но ни разу не использована (и external=False).\n"
                        f"FILE: {f.path}"
                    )

        return project

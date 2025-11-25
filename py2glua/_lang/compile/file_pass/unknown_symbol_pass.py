from __future__ import annotations

import builtins

from ...py.py_ir_dataclass import (
    PyIRAssign,
    PyIRClassDef,
    PyIRFile,
    PyIRFunctionDef,
    PyIRVarUse,
)
from ..compile_exception import CompileError


class UnknownSymbolPass:
    @classmethod
    def run(cls, file: PyIRFile):
        defined: set[str] = set()

        defined.update(dir(builtins))

        defined.add("__realm__")

        ctx = getattr(file, "context", None)
        if ctx is not None:
            imports = ctx.meta.get("imports", [])
            for mod in imports:
                last = mod.split(".")[-1]
                if last:
                    defined.add(last)

            aliases = ctx.meta.get("aliases", {})
            for alias_name in aliases.keys():
                defined.add(alias_name)

        for node in file.body:
            if isinstance(node, PyIRAssign):
                for t in node.targets:
                    name = getattr(t, "name", None)
                    if name is not None:
                        defined.add(name)

            elif isinstance(node, PyIRFunctionDef):
                defined.add(node.name)

            elif isinstance(node, PyIRClassDef):
                defined.add(node.name)

        for node in file.walk():
            if isinstance(node, PyIRVarUse):
                name = node.name

                if name not in defined:
                    raise CompileError(
                        f"Unknown symbol '{name}'\n"
                        f"LINE|OFFSET: {node.line}|{node.offset}\n"
                        f"FILE: {file.path}"
                    )

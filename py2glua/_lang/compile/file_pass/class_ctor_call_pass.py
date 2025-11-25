from __future__ import annotations

from ...py.py_ir_dataclass import (
    PyIRCall,
    PyIRClassDef,
    PyIRFile,
    PyIRFunctionDef,
    PyIRVarUse,
)
from ..compiler import Compiler


class ClassCtorCallPass:
    @staticmethod
    def run(file: PyIRFile, compiler: Compiler) -> PyIRFile:
        class_ctors: dict[str, str] = {}

        for node in file.body:
            if isinstance(node, PyIRClassDef):
                ctor = "__init__"

                for stmt in node.body:
                    if isinstance(stmt, PyIRFunctionDef):
                        if stmt.name in (
                            "__init__",
                        ) or stmt.name == compiler.config.get("method_renames", {}).get(
                            "__init__"
                        ):
                            ctor = stmt.name
                            break

                class_ctors[node.name] = ctor

        if not class_ctors:
            return file

        for node in file.walk():
            if isinstance(node, PyIRCall):
                if node.name in class_ctors:
                    ClassCtorCallPass._rewrite_call(node, class_ctors[node.name])

        return file

    @staticmethod
    def _rewrite_call(node: PyIRCall, ctor_name: str):
        class_name = node.name
        node.name = f"{class_name}.{ctor_name}"

        node.args_p.insert(0, PyIRVarUse(node.line, node.offset, class_name))

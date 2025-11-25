from __future__ import annotations

from typing import Dict

from ...py.py_ir_dataclass import (
    PyIRClassDef,
    PyIRFile,
    PyIRFunctionDef,
)
from ..compiler import Compiler


class ClassMethodRenamePass:
    @staticmethod
    def run(file: PyIRFile, compiler: Compiler) -> PyIRFile:
        mapping: Dict[str, str] = compiler.config.get("method_renames", {})
        if not mapping:
            return file

        for node in file.body:
            if isinstance(node, PyIRClassDef):
                for stmt in node.body:
                    if isinstance(stmt, PyIRFunctionDef):
                        old = stmt.name
                        if old in mapping:
                            stmt.name = mapping[old]

        return file

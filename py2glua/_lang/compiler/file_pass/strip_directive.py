from __future__ import annotations

from pathlib import Path

from ...py.ir_builder import PyIRFile
from ...py.ir_dataclass import (
    PyIRClassDef,
    PyIRComment,
    PyIRFunctionDef,
    PyIRNode,
)


class StripDirectivePass:
    """
    Обрабатывает:
      - специальные классы-директивы компилятора

    Целевые классы:
      - CompilerDirective
      - InternalCompilerDirective

    Назначение:
      - удаляет Python-реализацию директив
      - оставляет только сигнатуры
      - удаляет декораторы с методов
    """

    TARGET_CLASSES = {
        "CompilerDirective",
        "InternalCompilerDirective",
    }

    @classmethod
    def run(cls, ir: PyIRFile) -> PyIRFile:
        cls._process_block(ir.body, ir.path)
        return ir

    @classmethod
    def _process_block(
        cls,
        body: list[PyIRNode],
        file_path: Path | None,
    ) -> None:
        for node in body:
            if isinstance(node, PyIRClassDef):
                if node.name in cls.TARGET_CLASSES:
                    cls._strip_class(node)

                else:
                    cls._process_block(node.body, file_path)

    @classmethod
    def _strip_class(cls, node: PyIRClassDef) -> None:
        cls._strip_docstring(node.body)

        for item in node.body:
            if isinstance(item, PyIRFunctionDef):
                cls._strip_function(item)

    @staticmethod
    def _strip_function(fn: PyIRFunctionDef) -> None:
        StripDirectivePass._strip_docstring(fn.body)

        fn.body.clear()
        fn.decorators.clear()

    @staticmethod
    def _strip_docstring(body: list[PyIRNode]) -> None:
        if not body:
            return

        first = body[0]
        if isinstance(first, PyIRComment):
            body.pop(0)

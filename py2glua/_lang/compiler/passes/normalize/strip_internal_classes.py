from __future__ import annotations

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRClassDef,
    PyIRComment,
    PyIRFunctionDef,
    PyIRNode,
)


class StripInternalClassesPass:
    """
    Нормализация internal-классов:
    - выкинуть комментарии из тела класса
    - занулить реализации всех методов

    Это применяется независимо от декораторов.
    """

    INTERNAL_CLASSES: set[str] = {
        "InternalCompilerDirective",
        "CompilerDirective",
    }

    @classmethod
    def run(cls, ir: PyIRFile) -> PyIRFile:
        cls._process_block(ir.body)
        return ir

    @classmethod
    def _process_block(cls, body: list[PyIRNode]) -> None:
        for node in body:
            if isinstance(node, PyIRClassDef):
                if node.name in cls.INTERNAL_CLASSES:
                    cls._strip_class(node)

                cls._process_block(node.body)
                continue

            if isinstance(node, PyIRFunctionDef):
                cls._process_block(node.body)
                continue

    @classmethod
    def _strip_class(cls, cls_node: PyIRClassDef) -> None:
        new_body: list[PyIRNode] = []

        for item in cls_node.body:
            if isinstance(item, PyIRComment):
                continue

            if isinstance(item, PyIRFunctionDef):
                item.body = []

            new_body.append(item)

        cls_node.body = new_body

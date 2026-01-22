from __future__ import annotations

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRClassDef,
    PyIRFunctionDef,
    PyIRNode,
)


class StripBodiesByDecoratorPass:
    """
    Нормализация "stream body": стрипает тела по декораторам.

    Правила:
    - декоратор на функции/методе -> стрипает ТОЛЬКО эту функцию
    - декоратор на классе -> стрипает тела ВСЕХ методов класса
    """

    STRIP_DECORATORS: set[str] = {
        "CompilerDirective.no_compile",
        "CompilerDirective.gmod_api",
    }

    @classmethod
    def run(cls, ir: PyIRFile) -> PyIRFile:
        cls._process_block(ir.body)
        return ir

    @classmethod
    def _process_block(cls, body: list[PyIRNode]) -> None:
        for node in body:
            if isinstance(node, PyIRFunctionDef):
                if cls._has_strip_decorator(node):
                    node.body = []

                cls._process_block(node.body)
                continue

            if isinstance(node, PyIRClassDef):
                if cls._has_strip_decorator(node):
                    cls._strip_all_methods(node)
                else:
                    cls._strip_marked_methods(node)

                cls._process_block(node.body)
                continue

    @classmethod
    def _strip_all_methods(cls, cls_node: PyIRClassDef) -> None:
        for item in cls_node.body:
            if isinstance(item, PyIRFunctionDef):
                item.body = []

    @classmethod
    def _strip_marked_methods(cls, cls_node: PyIRClassDef) -> None:
        for item in cls_node.body:
            if isinstance(item, PyIRFunctionDef) and cls._has_strip_decorator(item):
                item.body = []

    @classmethod
    def _has_strip_decorator(cls, node) -> bool:
        for d in node.decorators:
            name = d.name
            if name in cls.STRIP_DECORATORS:
                return True

        return False

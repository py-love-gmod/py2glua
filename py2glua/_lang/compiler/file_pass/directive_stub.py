from __future__ import annotations

from ...py.ir_builder import PyIRFile
from ...py.ir_dataclass import PyIRClassDef, PyIRComment, PyIRFunctionDef, PyIRNode


class DirectiveStubPass:
    NO_COMPILE_NAMES = {
        "no_compile",
        "InternalCompilerDirective.no_compile",
        "gmod_api",
        "InternalCompilerDirective.gmod_api",
    }

    @classmethod
    def run(cls, ir: PyIRFile) -> PyIRFile:
        cls._process_block(ir.body)
        return ir

    @classmethod
    def _process_block(cls, body: list[PyIRNode]) -> None:
        for node in body:
            if isinstance(node, PyIRFunctionDef):
                cls._process_function(node)

            elif isinstance(node, PyIRClassDef):
                cls._process_class(node)

    @classmethod
    def _process_function(cls, fn: PyIRFunctionDef) -> None:
        has_no_compile = cls._has_decorator(fn, cls.NO_COMPILE_NAMES)

        if has_no_compile:
            cls._strip_body(fn.body)

    @classmethod
    def _process_class(cls, cls_node: PyIRClassDef) -> None:
        has_no_compile = cls._has_decorator(cls_node, cls.NO_COMPILE_NAMES)

        if has_no_compile:
            cls._strip_class_body(cls_node.body)

            for item in cls_node.body:
                if isinstance(item, PyIRFunctionDef):
                    cls._strip_body(item.body)

        cls._process_block(cls_node.body)

    @staticmethod
    def _has_decorator(
        node: PyIRFunctionDef | PyIRClassDef,
        names: set[str],
    ) -> bool:
        return any(d.name in names for d in node.decorators)

    @staticmethod
    def _remove_decorators(
        node: PyIRFunctionDef | PyIRClassDef,
        names: set[str],
    ) -> None:
        node.decorators[:] = [d for d in node.decorators if d.name not in names]

    @staticmethod
    def _strip_body(body: list[PyIRNode]) -> None:
        body[:] = [n for n in body if not isinstance(n, PyIRComment)]
        body.clear()

    @staticmethod
    def _strip_class_body(body: list[PyIRNode]) -> None:
        body[:] = [n for n in body if not isinstance(n, PyIRComment)]

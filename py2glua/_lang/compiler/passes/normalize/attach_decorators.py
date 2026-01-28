from __future__ import annotations

from pathlib import Path

from ....._cli.logging_setup import exit_with_code
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRClassDef,
    PyIRDecorator,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRWhile,
    PyIRWith,
)


class AttachDecoratorsPass:
    """
    Обрабатывает PyIRDecorator

    Назначение:
      - привязывает декораторы к следующему FunctionDef или ClassDef
      - работает рекурсивно во всех вложенных блоках
      - запрещает декораторы перед чем-либо кроме class/function
    """

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
        pending: list[PyIRDecorator] = []
        new_body: list[PyIRNode] = []

        for node in body:
            if isinstance(node, PyIRDecorator):
                pending.append(node)
                continue

            if isinstance(node, (PyIRFunctionDef, PyIRClassDef)):
                if pending:
                    node.decorators = pending + node.decorators
                    pending = []

                cls._process_block(node.body, file_path)
                new_body.append(node)
                continue

            if isinstance(node, PyIRIf):
                if pending:
                    cls._decorator_error(pending[0], file_path)

                cls._process_block(node.body, file_path)
                cls._process_block(node.orelse, file_path)
                new_body.append(node)
                continue

            if isinstance(node, PyIRFor):
                if pending:
                    cls._decorator_error(pending[0], file_path)

                cls._process_block(node.body, file_path)
                new_body.append(node)
                continue

            if isinstance(node, PyIRWhile):
                if pending:
                    cls._decorator_error(pending[0], file_path)

                cls._process_block(node.body, file_path)
                new_body.append(node)
                continue

            if isinstance(node, PyIRWith):
                if pending:
                    cls._decorator_error(pending[0], file_path)

                cls._process_block(node.body, file_path)
                new_body.append(node)
                continue

            if pending:
                cls._decorator_error(pending[0], file_path)

            new_body.append(node)

        if pending:
            cls._decorator_error(pending[0], file_path)

        body[:] = new_body

    @staticmethod
    def _decorator_error(dec: PyIRDecorator, file_path: Path | None) -> None:
        exit_with_code(
            1,
            "Декоратор не может быть применён к этому выражению\n"
            f"Файл: {file_path}\n"
            f"Строка: {dec.line}, позиция: {dec.offset}",
        )
        raise AssertionError("unreachable")

from __future__ import annotations

from pathlib import Path

from ....._cli import CompilerExit
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRClassDef,
    PyIRComment,
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

            if isinstance(node, PyIRComment):
                new_body.append(node)
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
                    CompilerExit.user_error_node(
                        "Декоратор не может быть применён к этому выражению",
                        file_path,
                        pending[0],
                    )

                cls._process_block(node.body, file_path)
                cls._process_block(node.orelse, file_path)
                new_body.append(node)
                continue

            if isinstance(node, PyIRFor):
                if pending:
                    CompilerExit.user_error_node(
                        "Декоратор не может быть применён к этому выражению",
                        file_path,
                        pending[0],
                    )

                cls._process_block(node.body, file_path)
                new_body.append(node)
                continue

            if isinstance(node, PyIRWhile):
                if pending:
                    CompilerExit.user_error_node(
                        "Декоратор не может быть применён к этому выражению",
                        file_path,
                        pending[0],
                    )

                cls._process_block(node.body, file_path)
                new_body.append(node)
                continue

            if isinstance(node, PyIRWith):
                if pending:
                    CompilerExit.user_error_node(
                        "Декоратор не может быть применён к этому выражению",
                        file_path,
                        pending[0],
                    )

                cls._process_block(node.body, file_path)
                new_body.append(node)
                continue

            if pending:
                CompilerExit.user_error_node(
                    "Декоратор не может быть применён к этому выражению",
                    file_path,
                    pending[0],
                )

            new_body.append(node)

        if pending:
            CompilerExit.user_error_node(
                "Декоратор не может быть применён к этому выражению",
                file_path,
                pending[0],
            )

        body[:] = new_body

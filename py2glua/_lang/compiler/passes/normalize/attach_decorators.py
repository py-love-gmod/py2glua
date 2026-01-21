from __future__ import annotations

from pathlib import Path

from ....._cli.logging_setup import exit_with_code
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRClassDef,
    PyIRDecorator,
    PyIRFunctionDef,
    PyIRNode,
)


class AttachDecoratorsPass:
    """
    Обрабатывает:
      - PyIRDecorator на уровне файла, классов и функций

    Назначение:
      - привязывает декораторы к следующему объявлению
        (классу или функции)
      - поддерживает вложенные блоки

    Правила:
      - декоратор применяется только к классу или функции
      - декоратор обязан стоять непосредственно перед объявлением
      - несколько декораторов подряд накапливаются
      - после применения декораторы удаляются из тела блока
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

            if isinstance(node, (PyIRClassDef, PyIRFunctionDef)):
                if pending:
                    node.decorators = pending + node.decorators
                    pending = []

                cls._process_block(node.body, file_path)
                new_body.append(node)
                continue

            if pending:
                d = pending[0]
                exit_with_code(
                    1,
                    "Декоратор не может быть применён к этому выражению\n"
                    f"Файл: {file_path}\n"
                    f"Строка: {d.line}, позиция: {d.offset}",
                )
                raise AssertionError("unreachable")

            new_body.append(node)

        if pending:
            d = pending[0]
            exit_with_code(
                1,
                "Декоратор не был применён ни к одной функции или классу\n"
                f"Файл: {file_path}\n"
                f"Строка: {d.line}, позиция: {d.offset}",
            )
            raise AssertionError("unreachable")

        body[:] = new_body

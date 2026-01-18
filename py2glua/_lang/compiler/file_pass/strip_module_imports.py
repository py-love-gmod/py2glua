from __future__ import annotations

from ...py.ir_builder import PyIRFile
from ...py.ir_dataclass import PyIRImport, PyIRNode


class StripModulesImportPass:
    """
    Обрабатывает:
      - import / from-import указанных модулей

    Назначение:
      - полностью удаляет импорты заданных модулей
      - используется для очистки Python-only зависимостей
        (typing, collections.abc, dataclasses и т.п.)

    Примечание:
      - pass не анализирует использование
      - ответственность за корректность лежит на вызывающем
    """

    TARGET_MODULES: set[tuple[str, ...]] = {
        ("typing",),
        ("collections", "abc"),
        ("pathlib",),
        ("warnings",),
    }

    @classmethod
    def run(cls, ir: PyIRFile) -> PyIRFile:
        ir.body[:] = cls._filter_block(ir.body)
        return ir

    @classmethod
    def _filter_block(cls, body: list[PyIRNode]) -> list[PyIRNode]:
        out: list[PyIRNode] = []

        for node in body:
            if isinstance(node, PyIRImport) and cls._should_strip(node):
                continue

            out.append(node)

        return out

    @classmethod
    def _should_strip(cls, imp: PyIRImport) -> bool:
        if not imp.modules:
            return False

        modules = tuple(imp.modules)

        return modules in cls.TARGET_MODULES

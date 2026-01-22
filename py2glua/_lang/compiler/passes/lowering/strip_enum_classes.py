from __future__ import annotations

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import PyIRClassDef, PyIRNode
from ..analysis.ctx import AnalysisContext, AnalysisPass


class StripEnumClassesPass(AnalysisPass):
    """
    Удаляет enum-классы после анализа:
    - python enum.Enum
    - @CompilerDirective.gmod_special_enum

    Используется ТОЛЬКО после CollectEnumsPass.
    """

    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if not ctx.enums:
            return

        new_body: list[PyIRNode] = []

        for node in ir.body:
            if isinstance(node, PyIRClassDef):
                sid = cls._get_class_symbol_id(node, ctx)
                if sid is not None and sid in ctx.enums:
                    continue

            new_body.append(node)

        ir.body = new_body

    @staticmethod
    def _get_class_symbol_id(
        node: PyIRClassDef,
        ctx: AnalysisContext,
    ) -> int | None:
        for sid, info in ctx.symbols.items():
            if info.symbol.kind == "class" and info.symbol.name == node.name:
                return sid

        return None

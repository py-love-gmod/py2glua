from __future__ import annotations

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import PyIRClassDef
from .ctx import AnalysisContext, AnalysisPass


class CollectWithConditionClassesPass(AnalysisPass):
    """
    Собирает классы, помеченные @CompilerDirective.with_condition().
    Используется для последующего lowering конструкции `with <Class>.<attr>:`.
    """

    TARGET_DECORATOR = "CompilerDirective.with_condition"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if ir.path is None:
            return

        for node in ir.body:
            if not isinstance(node, PyIRClassDef):
                continue

            if not cls._has_with_condition_decorator(node):
                continue

            sid = cls._get_class_symbol_id(node.name, ir, ctx)
            if sid is None:
                continue

            ctx.with_condition_classes.add(sid)

    @classmethod
    def _has_with_condition_decorator(cls, node: PyIRClassDef) -> bool:
        return any(d.name == cls.TARGET_DECORATOR for d in node.decorators)

    @staticmethod
    def _get_class_symbol_id(
        class_name: str,
        ir: PyIRFile,
        ctx: AnalysisContext,
    ) -> int | None:
        if ir.path is None:
            return None

        for sid, info in ctx.symbols.items():
            if info.file != ir.path:
                continue

            sym = info.symbol
            if sym.scope == "module" and sym.kind == "class" and sym.name == class_name:
                return sid

        return None

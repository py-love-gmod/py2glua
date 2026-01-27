from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import PyIRAttribute, PyIRIf, PyIRNode, PyIRWith
from ...ir_compiler import PyIRSymbolRef
from ..analysis.ctx import AnalysisContext


class LowerWithConditionPass:
    """
    Преобразует:

        with Class.attr:
            body

    в:

        if Class.attr:
            body

    если Class помечен @CompilerDirective.with_condition().
    """

    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if not ctx.with_condition_classes:
            return

        ir.body = cls._rewrite_block(ir.body, ctx)

    @classmethod
    def _rewrite_block(
        cls, body: list[PyIRNode], ctx: AnalysisContext
    ) -> list[PyIRNode]:
        out: list[PyIRNode] = []
        for n in body:
            out.append(cls._rewrite_node(n, ctx))

        return out

    @classmethod
    def _rewrite_node(cls, node: PyIRNode, ctx: AnalysisContext) -> PyIRNode:
        if isinstance(node, PyIRWith):
            cond = cls._try_extract_with_condition(node, ctx)
            if cond is not None:
                return PyIRIf(
                    line=node.line,
                    offset=node.offset,
                    test=cond,
                    body=cls._rewrite_block(node.body, ctx),
                    orelse=[],
                )

        if not is_dataclass(node):
            return node

        for f in fields(node):
            val = getattr(node, f.name)
            new_val = cls._rewrite_value(val, ctx)
            if new_val is not val:
                setattr(node, f.name, new_val)

        return node

    @classmethod
    def _rewrite_value(cls, val: Any, ctx: AnalysisContext) -> Any:
        if isinstance(val, PyIRNode):
            return cls._rewrite_node(val, ctx)

        if isinstance(val, list):
            return [cls._rewrite_value(x, ctx) for x in val]

        if isinstance(val, tuple):
            return tuple(cls._rewrite_value(x, ctx) for x in val)

        if isinstance(val, dict):
            return {
                cls._rewrite_value(k, ctx): cls._rewrite_value(v, ctx)
                for k, v in val.items()
            }

        return val

    @staticmethod
    def _try_extract_with_condition(
        node: PyIRWith, ctx: AnalysisContext
    ) -> PyIRNode | None:
        if len(node.items) != 1:
            return None

        item = node.items[0]
        if item.optional_vars is not None:
            return None

        expr = item.context_expr
        if not isinstance(expr, PyIRAttribute):
            return None

        base = expr.value
        if not isinstance(base, PyIRSymbolRef):
            return None

        if base.sym_id not in ctx.with_condition_classes:
            return None

        return expr

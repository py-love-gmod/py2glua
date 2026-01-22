from __future__ import annotations

from dataclasses import fields, is_dataclass

from ....._cli.logging_setup import exit_with_code
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRConstant,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRNode,
)
from ...ir_compiler import PyIRSymbolRef
from ..analysis.ctx import AnalysisContext, AnalysisPass


class EnumUsagePass(AnalysisPass):
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if not ctx.enums:
            return

        ir.body = cls._rewrite_block(ir.body, ctx, ir)

    @classmethod
    def _rewrite_block(
        cls,
        body: list[PyIRNode],
        ctx: AnalysisContext,
        ir: PyIRFile,
    ) -> list[PyIRNode]:
        return [cls._rewrite_node(n, ctx, ir) for n in body]

    @classmethod
    def _rewrite_node(
        cls,
        node: PyIRNode,
        ctx: AnalysisContext,
        ir: PyIRFile,
    ) -> PyIRNode:
        if isinstance(node, PyIRSymbolRef):
            if node.sym_id in ctx.enums:
                enum_info = ctx.enums[node.sym_id]
                cls._error(
                    ir=ir,
                    line=node.line,
                    offset=node.offset,
                    msg=(f"Enum '{enum_info.fqname}' используется как значение."),
                )
                assert False, "unreachable"

        if isinstance(node, PyIRAttribute):
            repl = cls._try_replace_enum_attr(node, ctx, ir)
            if repl is not None:
                return repl

        if not is_dataclass(node):
            return node

        for f in fields(node):
            val = getattr(node, f.name)
            new_val = cls._rewrite_value(val, ctx, ir)
            if new_val is not val:
                setattr(node, f.name, new_val)

        return node

    @classmethod
    def _rewrite_value(cls, val, ctx: AnalysisContext, ir: PyIRFile):
        if isinstance(val, PyIRNode):
            return cls._rewrite_node(val, ctx, ir)

        if isinstance(val, list):
            return [cls._rewrite_value(x, ctx, ir) for x in val]

        if isinstance(val, tuple):
            return tuple(cls._rewrite_value(x, ctx, ir) for x in val)

        if isinstance(val, dict):
            return {
                cls._rewrite_value(k, ctx, ir): cls._rewrite_value(v, ctx, ir)
                for k, v in val.items()
            }

        return val

    @classmethod
    def _try_replace_enum_attr(
        cls,
        node: PyIRAttribute,
        ctx: AnalysisContext,
        ir: PyIRFile,
    ) -> PyIRNode | None:
        if not isinstance(node.value, PyIRSymbolRef):
            return None

        enum_sid = node.value.sym_id
        enum_info = ctx.enums.get(enum_sid)
        if enum_info is None:
            return None

        value = enum_info.values.get(node.attr)
        if value is None:
            cls._error(
                ir=ir,
                line=node.line,
                offset=node.offset,
                msg=(
                    f"Enum '{enum_info.fqname}' не содержит поля '{node.attr}'.\n"
                    f"Доступные поля: {', '.join(enum_info.values.keys()) or '<пусто>'}"
                ),
            )
            assert False, "unreachable"

        if value.kind in ("int", "str", "bool"):
            return PyIRConstant(
                line=node.line,
                offset=node.offset,
                value=value.value,
            )

        if value.kind == "global":
            if not isinstance(value.value, str):
                cls._error(
                    ir=ir,
                    line=node.line,
                    offset=node.offset,
                    msg=(
                        f"Значение enum '{enum_info.fqname}.{value.name}' "
                        f"с типом 'global' должно быть строкой."
                    ),
                )
                assert False, "unreachable"

            return PyIREmitExpr(
                line=node.line,
                offset=node.offset,
                kind=PyIREmitKind.GLOBAL,
                name=value.value,
            )

        cls._error(
            ir=ir,
            line=node.line,
            offset=node.offset,
            msg=(
                f"Неизвестный тип значения enum '{value.kind}' "
                f"в '{enum_info.fqname}.{value.name}'."
            ),
        )

    @staticmethod
    def _error(*, ir: PyIRFile, line: int | None, offset: int | None, msg: str) -> None:
        full_msg = f"{msg}\nФайл: {ir.path}\nLINE|OFFSET: {line}|{offset}"

        exit_with_code(1, full_msg)
        assert False, "unreachable"

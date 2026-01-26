from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from ....._cli.logging_setup import exit_with_code
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import PyIRCall, PyIRNode, PyIRWith
from ...ir_compiler import PyIRSymbolRef
from ..analysis.ctx import AnalysisContext


class LowerContextManagersPass:
    """
    Раскрывает конструкции:

        with foo():
            body

    где foo - функция, помеченная @CompilerDirective.contextmanager.

    Подстановка:
        pre_body
        body
        post_body

    Если with не распознан - билд ломается.
    """

    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        ir.body = cls._rewrite_block(ir.body, ctx, ir)

    @classmethod
    def _rewrite_block(
        cls,
        body: list[PyIRNode],
        ctx: AnalysisContext,
        ir: PyIRFile,
    ) -> list[PyIRNode]:
        out: list[PyIRNode] = []

        for node in body:
            repl = cls._rewrite_node(node, ctx, ir)
            if isinstance(repl, list):
                out.extend(repl)
            else:
                out.append(repl)

        return out

    @classmethod
    def _rewrite_node(
        cls,
        node: PyIRNode,
        ctx: AnalysisContext,
        ir: PyIRFile,
    ) -> PyIRNode | list[PyIRNode]:
        if isinstance(node, PyIRWith):
            return cls._lower_with(node, ctx, ir)

        if not is_dataclass(node):
            return node

        for f in fields(node):
            val = getattr(node, f.name)
            new_val = cls._rewrite_value(val, ctx, ir)
            if new_val is not val:
                setattr(node, f.name, new_val)

        return node

    @classmethod
    def _rewrite_value(cls, val: Any, ctx: AnalysisContext, ir: PyIRFile):
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
    def _lower_with(
        cls,
        node: PyIRWith,
        ctx: AnalysisContext,
        ir: PyIRFile,
    ) -> list[PyIRNode]:
        if len(node.items) != 1:
            cls._error(
                ir,
                node,
                "Конструкция with с несколькими items не поддерживается.",
            )

        item = node.items[0]

        if item.optional_vars is not None:
            cls._error(
                ir,
                item,
                "Конструкция `with ... as x` не поддерживается.",
            )

        expr = item.context_expr

        if not isinstance(expr, PyIRCall):
            cls._error(
                ir,
                item,
                "Данная конструкция with не поддерживается.",
            )
            assert False

        func = expr.func
        if not isinstance(func, PyIRSymbolRef):
            cls._error(
                ir,
                item,
                "Данная конструкция with не поддерживается.",
            )
            assert False

        info = ctx.context_managers.get(func.sym_id)
        if info is None:
            cls._error(
                ir,
                item,
                "Используется with для функции, не помеченной @contextmanager.",
            )
            assert False

        out: list[PyIRNode] = []

        out.extend(cls._clone_block(info.pre_body))
        out.extend(cls._rewrite_block(node.body, ctx, ir))
        out.extend(cls._clone_block(info.post_body))

        return out

    @staticmethod
    def _clone_block(body: list[PyIRNode]) -> list[PyIRNode]:
        return list(body)

    @staticmethod
    def _error(ir: PyIRFile, node: PyIRNode, msg: str) -> None:
        full_msg = f"{msg}\nФайл: {ir.path}\nLINE|OFFSET: {node.line}|{node.offset}"
        exit_with_code(1, full_msg)
        assert False, "unreachable"

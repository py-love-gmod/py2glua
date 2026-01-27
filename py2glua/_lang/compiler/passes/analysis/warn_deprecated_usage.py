from __future__ import annotations

from ....._cli.logging_setup import logger
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import PyIRAttribute, PyIRNode
from ...ir_compiler import PyIRSymbolRef
from .ctx import AnalysisContext, AnalysisPass


def _fmt_msg(s: str | None) -> str:
    if not s:
        return "<без сообщения>"

    return s.replace("\\n", "\n")


class WarnDeprecatedUsagePass(AnalysisPass):
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if not ctx.deprecated_symbols and not ctx.deprecated_members:
            return

        for node in ir.walk():
            cls._check_node(node, ir=ir, ctx=ctx)

    @classmethod
    def _check_node(cls, node: PyIRNode, *, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if isinstance(node, PyIRSymbolRef):
            dep = ctx.deprecated_symbols.get(node.sym_id)
            if dep is None:
                return

            cls._warn(
                dep_fqname=dep.fqname,
                dep_msg=dep.message,
                node=node,
                ir=ir,
                ctx=ctx,
            )
            return

        if isinstance(node, PyIRAttribute) and isinstance(node.value, PyIRSymbolRef):
            key = (node.value.sym_id, node.attr)
            dep = ctx.deprecated_members.get(key)
            if dep is None:
                return

            cls._warn(
                dep_fqname=dep.fqname,
                dep_msg=dep.message,
                node=node,
                ir=ir,
                ctx=ctx,
            )
            return

    @classmethod
    def _warn(
        cls,
        *,
        dep_fqname: str,
        dep_msg: str | None,
        node: PyIRNode,
        ir: PyIRFile,
        ctx: AnalysisContext,
    ) -> None:
        loc_key = f"{dep_fqname}|{ir.path}|{node.line}|{node.offset}"
        dedup_key = (ir.path, node.line, node.offset, loc_key)
        if dedup_key in ctx.deprecated_warned:
            return

        ctx.deprecated_warned.add(dedup_key)

        msg = f"""Используется устаревший символ
Символ: {dep_fqname}
Сообщение: {_fmt_msg(dep_msg)}
Файл: {ir.path}
LINE|OFFSET: {node.line}|{node.offset}
        """

        logger.warning(msg)

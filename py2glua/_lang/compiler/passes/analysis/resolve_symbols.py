from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRNode,
    PyIRVarUse,
)
from ...ir_compiler import PyIRSymbolRef
from .ctx import AnalysisContext, AnalysisPass


class ResolveSymbolsPass(AnalysisPass):
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if not ctx.symbol_id_by_fqname and not ctx.symbol_ids_by_name:
            return

        ir.body = cls._rewrite_block(ir.body, ctx)

    @classmethod
    def _rewrite_block(
        cls, body: list[PyIRNode], ctx: AnalysisContext
    ) -> list[PyIRNode]:
        return [cls._rewrite_node(n, ctx) for n in body]

    @classmethod
    def _rewrite_node(cls, node: PyIRNode, ctx: AnalysisContext) -> PyIRNode:
        if isinstance(node, PyIRAttribute):
            return cls._rewrite_attribute_chain(node, ctx)

        if isinstance(node, PyIRVarUse):
            ids = ctx.symbol_ids_by_name.get(node.name)
            if ids and len(ids) == 1:
                return PyIRSymbolRef(line=node.line, offset=node.offset, sym_id=ids[0])

            return node

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
            changed = False
            out: list[Any] = []
            for x in val:
                nx = cls._rewrite_value(x, ctx)
                changed = changed or (nx is not x)
                out.append(nx)

            return out if changed else val

        if isinstance(val, dict):
            changed = False
            out_1: dict[Any, Any] = {}
            for k, v in val.items():
                nk = cls._rewrite_value(k, ctx)
                nv = cls._rewrite_value(v, ctx)
                changed = changed or (nk is not k) or (nv is not v)
                out_1[nk] = nv

            return out_1 if changed else val

        if isinstance(val, tuple):
            changed = False
            out: list[Any] = []
            for x in val:
                nx = cls._rewrite_value(x, ctx)
                changed = changed or (nx is not x)
                out.append(nx)

            return tuple(out) if changed else val

        return val

    @classmethod
    def _rewrite_attribute_chain(
        cls, node: PyIRAttribute, ctx: AnalysisContext
    ) -> PyIRNode:
        parts = cls._collect_dotted_parts(node)
        if parts is None:
            node.value = cls._rewrite_node(node.value, ctx)  # type: ignore[assignment]
            return node

        sid, used_len = cls._find_longest_symbol_prefix(parts, ctx)
        if sid is None:
            node.value = cls._rewrite_node(node.value, ctx)  # type: ignore[assignment]
            return node

        base = PyIRSymbolRef(line=node.line, offset=node.offset, sym_id=sid)

        rest = parts[used_len:]
        expr: PyIRNode = base
        for attr in rest:
            expr = PyIRAttribute(
                line=node.line, offset=node.offset, value=expr, attr=attr
            )

        return expr

    @staticmethod
    def _collect_dotted_parts(node: PyIRAttribute) -> list[str] | None:
        parts: list[str] = []
        cur: PyIRNode = node

        while isinstance(cur, PyIRAttribute):
            parts.append(cur.attr)
            cur = cur.value

        if isinstance(cur, PyIRVarUse):
            parts.append(cur.name)
            parts.reverse()
            return parts

        if isinstance(cur, PyIRSymbolRef):
            return None

        return None

    @staticmethod
    def _find_longest_symbol_prefix(
        parts: list[str], ctx: AnalysisContext
    ) -> tuple[int | None, int]:
        for k in range(len(parts), 0, -1):
            fq = ".".join(parts[:k])
            sid = ctx.symbol_id_by_fqname.get(fq)
            if sid is not None:
                return sid, k

        return None, 0

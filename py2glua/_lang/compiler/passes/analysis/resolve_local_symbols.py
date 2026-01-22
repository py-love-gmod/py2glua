from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRFunctionDef,
    PyIRNode,
    PyIRVarUse,
)
from ...ir_compiler import PyIRLocalRef
from .ctx import AnalysisContext, AnalysisPass


class _Scope:
    def __init__(self, parent: "_Scope | None", allow_locals: bool):
        self.parent = parent
        self.allow_locals = allow_locals
        self.locals: dict[str, int] = {}

    def resolve(self, name: str) -> int | None:
        cur: _Scope | None = self
        while cur is not None:
            if name in cur.locals:
                return cur.locals[name]

            cur = cur.parent

        return None


class ResolveLocalSymbolsPass(AnalysisPass):
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        next_local_id = 1

        def alloc() -> int:
            nonlocal next_local_id
            lid = next_local_id
            next_local_id += 1
            return lid

        def rewrite_node(node: PyIRNode, scope: _Scope) -> PyIRNode:
            if isinstance(node, PyIRFunctionDef):
                fn_scope = _Scope(parent=scope, allow_locals=True)

                for arg in node.signature.keys():
                    fn_scope.locals[arg] = alloc()

                node.body = [rewrite_node(n, fn_scope) for n in node.body]
                return node

            if isinstance(node, PyIRAssign) and scope.allow_locals:
                for tgt in node.targets:
                    if isinstance(tgt, PyIRVarUse):
                        if scope.resolve(tgt.name) is None:
                            scope.locals[tgt.name] = alloc()

                return rewrite_generic(node, scope)

            if isinstance(node, PyIRVarUse):
                lid = scope.resolve(node.name)
                if lid is not None:
                    return PyIRLocalRef(
                        line=node.line,
                        offset=node.offset,
                        local_id=lid,
                        name=node.name,
                    )
                return node

            return rewrite_generic(node, scope)

        def rewrite_generic(node: PyIRNode, scope: _Scope) -> PyIRNode:
            if not is_dataclass(node):
                return node

            for f in fields(node):
                val = getattr(node, f.name)
                new_val = rewrite_value(val, scope)
                if new_val is not val:
                    setattr(node, f.name, new_val)

            return node

        def rewrite_value(val: Any, scope: _Scope) -> Any:
            if isinstance(val, PyIRNode):
                return rewrite_node(val, scope)

            if isinstance(val, list):
                return [rewrite_value(x, scope) for x in val]

            if isinstance(val, tuple):
                return tuple(rewrite_value(x, scope) for x in val)

            if isinstance(val, dict):
                return {
                    rewrite_value(k, scope): rewrite_value(v, scope)
                    for k, v in val.items()
                }

            return val

        module_scope = _Scope(parent=None, allow_locals=False)
        ir.body = [rewrite_node(n, module_scope) for n in ir.body]

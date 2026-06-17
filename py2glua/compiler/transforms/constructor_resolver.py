from __future__ import annotations

from plg_reader import IRCall, IRClassDef, IRNode, IRTransformer

from ..dt import CompilationUnit
from ..symbol_table import Scope, SymbolRefIR


class ConstructorResolver:
    @classmethod
    def resolve(cls, unit: CompilationUnit) -> CompilationUnit:
        transformer = _ConstructorTransformer(unit.node_to_scope)
        for ir_file in unit.irs.values():
            new_body: list[IRNode] = []
            for stmt in ir_file.body:
                result = transformer.visit(stmt)
                if result is not None:
                    new_body.append(result)

            ir_file.body = new_body

        return unit


class _ConstructorTransformer(IRTransformer):
    def __init__(self, node_to_scope: dict[int, Scope]):
        super().__init__()
        self.node_to_scope = node_to_scope

    def visit_IRCall(self, node: IRCall) -> IRNode | None:
        result = self.generic_visit(node)
        if result is None or not isinstance(result, IRCall):
            return result

        node = result

        func = node.func
        if not isinstance(func, SymbolRefIR):
            return node

        sym = func.symbol
        if sym.kind != "class":
            return node

        class_def = sym.node
        if not isinstance(class_def, IRClassDef):
            return node

        class_scope = self.node_to_scope.get(id(class_def))
        if class_scope is None:
            return node

        init_sym = class_scope.symbols.get("__init__")
        if init_sym is None:
            return node

        new_func = SymbolRefIR(pos=node.pos, symbol=init_sym)
        init_sym.references.append(new_func)
        node.func = new_func
        return node

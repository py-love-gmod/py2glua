import ast

from ..glua import TYPE_REGISTRY
from .assign_emitter import AssignEmitter
from .base_emitter import NodeEmitter
from .return_emitter import ReturnEmitter


class FunctionEmitter(NodeEmitter):
    def __init__(self):
        super().__init__()
        self.assign = AssignEmitter()
        self.ret = ReturnEmitter()

    def emit(self, node: ast.FunctionDef, indent=0) -> str:
        args = ", ".join(a.arg for a in node.args.args)
        lines = []

        lines.extend(self._generate_type_checks(node, indent + 1))

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                lines.append(self.assign.emit(stmt, indent + 1))

            elif isinstance(stmt, ast.Return):
                lines.append(self.ret.emit(stmt, indent + 1))

        body = "\n".join(lines)
        return f"{self._indent(indent)}local function {node.name}({args})\n{body}\n{self._indent(indent)}end"

    def _generate_type_checks(self, node: ast.FunctionDef, indent: int) -> list[str]:
        checks = []
        for arg in node.args.args:
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                typename = arg.annotation.id
                cls = TYPE_REGISTRY.get(typename)
                if cls:
                    checks.append(self._indent(indent) + cls.lua_check(arg.arg))

        return checks

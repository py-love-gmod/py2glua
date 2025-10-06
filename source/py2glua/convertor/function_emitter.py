import ast
from .base_emitter import NodeEmitter
from .assign_emitter import AssignEmitter
from .return_emitter import ReturnEmitter

class FunctionEmitter(NodeEmitter):
    def __init__(self):
        super().__init__()
        self.assign = AssignEmitter()
        self.ret = ReturnEmitter()

    def emit(self, node: ast.FunctionDef, indent=0) -> str:
        args = ", ".join(a.arg for a in node.args.args)
        lines = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                lines.append(self.assign.emit(stmt, indent + 1))

            elif isinstance(stmt, ast.Return):
                lines.append(self.ret.emit(stmt, indent + 1))

        body = "\n".join(lines)
        return f"{self._indent(indent)}function {node.name}({args})\n{body}\n{self._indent(indent)}end"

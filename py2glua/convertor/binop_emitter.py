import ast

from .base_emitter import NodeEmitter
from ..exceptions import CompileError


class BinOpEmitter(NodeEmitter):
    OPS = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
    }

    def emit(self, node: ast.BinOp, indent=0) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        op = self.OPS.get(type(node.op), None)
        if op is None:
            raise CompileError(
                f"Unsupported binary operator '{type(node.op).__name__}' "
                f"at line {getattr(node, 'lineno', '?')}"
            )

        return f"{left} {op} {right}"

    def _expr(self, node):
        if isinstance(node, ast.BinOp):
            return self.emit(node)

        default = self._default_for_expr(node)
        if default is not None:
            return default

        else:
            raise NotImplementedError(f"Unsupported BinOpEmitter: {type(node)}")

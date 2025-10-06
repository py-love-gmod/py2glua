import ast

from .base_emitter import NodeEmitter
from .binop_emitter import BinOpEmitter


class ReturnEmitter(NodeEmitter):
    def __init__(self):
        super().__init__()
        self.binop = BinOpEmitter()

    def emit(self, node: ast.Return, indent=0) -> str:
        value = self._expr(node.value)
        return f"{self._indent(indent)}return {value}"

    def _expr(self, node):
        if isinstance(node, ast.BinOp):
            return self.binop.emit(node)

        default = self._default_for_expr(node)
        if default:
            return default

        else:
            raise NotImplementedError(f"Unsupported ReturnEmitter: {type(node)}")

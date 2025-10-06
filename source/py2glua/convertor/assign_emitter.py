import ast

from .base_emitter import NodeEmitter
from .binop_emitter import BinOpEmitter


class AssignEmitter(NodeEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.binop = BinOpEmitter()

    def emit(self, node: ast.Assign, indent: int = 0) -> str:
        target = node.targets[0].id  # type: ignore
        value = self._expr(node.value)
        return f"{self._indent(indent)}local {target} = {value}"

    def _expr(self, node):
        if isinstance(node, ast.BinOp):
            return self.binop.emit(node)

        default = self._default_for_expr(node)
        if default:
            return default

        else:
            raise NotImplementedError(f"Unsupported AssignEmitter: {type(node)}")

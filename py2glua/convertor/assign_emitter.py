import ast

from ..glua import Global
from .base_emitter import NodeEmitter
from .binop_emitter import BinOpEmitter


class AssignEmitter(NodeEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.binop = BinOpEmitter()

    def emit(self, node: ast.Assign | ast.AnnAssign, indent: int = 0) -> str:
        if isinstance(node, ast.Assign):
            return self._emit_assign(node, indent)

        if isinstance(node, ast.AnnAssign):
            return self._emit_ann_assign(node, indent)

        raise TypeError(f"Unsupported node for AssignEmitter: {type(node)}")

    def _emit_assign(self, node: ast.Assign, indent: int) -> str:
        target = self._target_name(node.targets[0])
        value_node = node.value
        prefix = "local "

        is_global, inner = Global.unwrap_mark_var_call(value_node)
        if is_global:
            prefix = ""
            if inner is None:
                raise ValueError("Global.mark_var requires a value")
            value_node = inner

        value = self._expr(value_node)
        return f"{self._indent(indent)}{prefix}{target} = {value}"

    def _emit_ann_assign(self, node: ast.AnnAssign, indent: int) -> str:
        target = self._target_name(node.target)
        prefix = (
            ""
            if node.annotation and Global.is_global_annotation(node.annotation)
            else "local "
        )
        value_node = node.value
        value = self._expr(value_node)
        return f"{self._indent(indent)}{prefix}{target} = {value}"

    def _expr(self, node):
        if node is None:
            return "nil"

        if isinstance(node, ast.BinOp):
            return self.binop.emit(node)

        default = self._default_for_expr(node)
        if default is not None:
            return default

        raise NotImplementedError(f"Unsupported AssignEmitter: {type(node)}")

    @staticmethod
    def _target_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id

        raise NotImplementedError(f"Unsupported assign target: {type(node)}")

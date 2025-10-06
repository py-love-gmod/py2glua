import ast


class NodeEmitter:
    def emit(self, node, indent: int = 0) -> str:
        raise NotImplementedError("emit() not implemented")

    def _indent(self, level: int) -> str:
        return "    " * level

    def _default_for_expr(self, node) -> None | str:
        if isinstance(node, ast.Constant):
            return repr(node.value)

        elif isinstance(node, ast.Name):
            return node.id

        else:
            return None

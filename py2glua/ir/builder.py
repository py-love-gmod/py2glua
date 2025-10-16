import ast
from collections.abc import Callable

from .ir_base import File, Import, ImportType, VarStore


class IRBuilder:
    func_reg = {}

    @classmethod
    def _get_func(cls, ast_node: ast.AST) -> Callable:
        name = type(ast_node).__name__
        func = getattr(cls, f"build_{name}", None)
        if func is None:
            raise NotImplementedError(f"Unsupported AST node: {name}")

        return func

    @classmethod
    def build_ir(cls, module: ast.Module):
        file = File(None, None)
        for stmt in module.body:
            ir_nodes = cls.build_node(stmt)
            if not ir_nodes:
                continue

            if isinstance(ir_nodes, list):
                file.body.extend(ir_nodes)

            else:
                file.body.append(ir_nodes)

        return file

    @classmethod
    def build_node(cls, node: ast.AST):
        if node is None:
            return None

        handler = getattr(cls, f"build_{type(node).__name__}", None)
        if handler is None:
            raise NotImplementedError(f"Unsupported node: {type(node).__name__}")

        return handler(node)

    # region build func

    # region import
    @staticmethod
    def build_Import(node: ast.Import) -> Import:
        names = [alias.name for alias in node.names]
        aliases = [alias.asname for alias in node.names]
        return Import(
            module=None,
            names=names,
            aliases=aliases,
            import_type=ImportType.UNKNOWN,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    @staticmethod
    def build_ImportFrom(node: ast.ImportFrom) -> Import:
        names = [alias.name for alias in node.names]
        aliases = [alias.asname for alias in node.names]
        return Import(
            module=node.module,
            names=names,
            aliases=aliases,
            import_type=ImportType.UNKNOWN,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    # endregion

    # region assign
    @staticmethod
    def build_Assign(node: ast.Assign) -> list[VarStore]:
        pass

    @staticmethod
    def build_AnnAssign(node: ast.AnnAssign) -> VarStore:
        pass

    @staticmethod
    def build_AugAssign(node: ast.AugAssign) -> list[VarStore]:
        pass

    # endregion

    # endregion

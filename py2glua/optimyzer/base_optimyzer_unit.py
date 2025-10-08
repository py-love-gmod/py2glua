import ast


class BaseOptimyzerUnit:
    @classmethod
    def optimize(cls, node: ast.AST, context: dict) -> ast.AST:
        raise NotImplementedError

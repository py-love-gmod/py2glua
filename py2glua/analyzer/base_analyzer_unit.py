import ast


class BaseAnalyzerUnit:
    @classmethod
    def analyze(cls, node: ast.AST, context: dict) -> dict:
        raise NotImplementedError

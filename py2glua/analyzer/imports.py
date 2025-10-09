import ast

from ..managers import AnalyzerManager, BaseAnalyzerUnit


@AnalyzerManager.register(ast.ImportFrom)
class ImportFromAnalyzer(BaseAnalyzerUnit[ast.ImportFrom]):
    def process(self, node: ast.ImportFrom, ctx: dict) -> None:
        module = node.module or ""
        for alias in node.names:
            full = f"{module}.{alias.name}"
            asname = alias.asname or alias.name
            ctx.setdefault("aliases", {})[asname] = full


@AnalyzerManager.register(ast.Import)
class ImportAnalyzer(BaseAnalyzerUnit[ast.Import]):
    def process(self, node: ast.Import, ctx: dict) -> None:
        for alias in node.names:
            full = alias.name
            asname = alias.asname or alias.name
            ctx.setdefault("aliases", {})[asname] = full

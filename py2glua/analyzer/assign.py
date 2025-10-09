import ast
import logging

from ..managers import AnalyzerManager, BaseAnalyzerUnit
from ..utils import is_global_call

logger = logging.getLogger("py2glua")


@AnalyzerManager.register(ast.Assign)
class AssignAnalyzer(BaseAnalyzerUnit[ast.Assign]):
    def process(self, node: ast.Assign, ctx: dict) -> None:
        aliases = ctx.get("aliases", {})

        if is_global_call(node.value, aliases, "var"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    setattr(target, "__py2glua_global__", True)


@AnalyzerManager.register(ast.AnnAssign)
class AnnAssignAnalyzer(BaseAnalyzerUnit[ast.AnnAssign]):
    def process(self, node: ast.AnnAssign, ctx: dict) -> None:
        aliases = ctx.get("aliases", {})

        # Не работать с пустыми анатациями типов
        if node.value and is_global_call(node.value, aliases, "var"):
            if isinstance(node.target, ast.Name):
                setattr(node.target, "__py2glua_global__", True)

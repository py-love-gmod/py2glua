import ast
import logging

from ..managers import AnalyzerManager, BaseAnalyzerUnit
from ..utils import is_global_call

logger = logging.getLogger("py2glua")


@AnalyzerManager.register(ast.FunctionDef)
class FunctionAnalyzer(BaseAnalyzerUnit[ast.FunctionDef]):
    def process(self, node: ast.FunctionDef, ctx: dict) -> None:
        aliases = ctx.get("aliases", {})

        for _, deco in enumerate(node.decorator_list):
            if is_global_call(deco, aliases, "func"):
                setattr(node, "__py2glua_global__", True)
                node.decorator_list = []
                break

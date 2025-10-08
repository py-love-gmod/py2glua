import ast

from py2glua.analyzer import AnalyzerManager
from py2glua.analyzer.base_analyzer_unit import BaseAnalyzerUnit


def test_analyzer_manager_register():
    @AnalyzerManager.register(ast.Assign)
    class TestAnalyzerUnit(BaseAnalyzerUnit): ...

    assert isinstance(AnalyzerManager.get(ast.parse("x = 1").body[0]), BaseAnalyzerUnit)

import ast

import pytest

from py2glua.optimyzer.base_optimyzer_unit import BaseOptimyzerUnit
from py2glua.optimyzer.optimization_level import OptimizationLevel
from py2glua.optimyzer.optimyzer_manager import OptimyzerManager


class DummyUnit(BaseOptimyzerUnit):
    @classmethod
    def optimize(cls, node: ast.AST, context: dict) -> ast.AST:
        return ast.AST()


@pytest.fixture(autouse=True)
def clean_manager():
    OptimyzerManager._optimzer_units.clear()
    yield
    OptimyzerManager._optimzer_units.clear()


def test_register_and_get_basic():
    @OptimyzerManager.register(ast.Assign, OptimizationLevel.O1)
    class AssignUnit(DummyUnit):
        pass

    node = ast.parse("x = 0").body[0]
    result = OptimyzerManager.get(OptimizationLevel.O1, node)
    assert isinstance(result, AssignUnit)
    assert isinstance(result.optimize(node, {}), ast.AST)


def test_register_multiple_levels_and_nodes():
    @OptimyzerManager.register(ast.Assign, OptimizationLevel.O2)
    class AssignO2(DummyUnit): ...

    @OptimyzerManager.register(ast.If, OptimizationLevel.O3)
    class IfO3(DummyUnit): ...

    assign_node = ast.parse("x = 1").body[0]
    if_node = ast.parse("if x: y\nelse: z").body[0]

    result_assign = OptimyzerManager.get(OptimizationLevel.O2, assign_node)
    result_if = OptimyzerManager.get(OptimizationLevel.O3, if_node)

    assert isinstance(result_assign, AssignO2)
    assert isinstance(result_if, IfO3)


def test_get_returns_none_for_unregistered_node():
    node = ast.parse("x").body[0]
    result = OptimyzerManager.get(OptimizationLevel.O1, node)
    assert result is None


def test_get_returns_none_for_unregistered_level():
    @OptimyzerManager.register(ast.Assign, OptimizationLevel.O1)
    class AssignUnit(DummyUnit): ...

    node = ast.parse("x = 0").body[0]
    result = OptimyzerManager.get(OptimizationLevel.O3, node)
    assert result is None


def test_register_creates_nested_dict_if_missing():
    @OptimyzerManager.register(ast.Expr, OptimizationLevel.O2)
    class ExprUnit(DummyUnit): ...

    assert OptimizationLevel.O2 in OptimyzerManager._optimzer_units
    assert ast.Expr in OptimyzerManager._optimzer_units[OptimizationLevel.O2]

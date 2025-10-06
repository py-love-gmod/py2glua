import ast

import pytest

from py2glua.convertor.assign_emitter import AssignEmitter


def test_simple_assign_constant():
    node = ast.Assign(
        targets=[ast.Name(id="x", ctx=ast.Store())],
        value=ast.Constant(value=42),
    )

    e = AssignEmitter()
    result = e.emit(node)
    assert result.strip() == "local x = 42"


def test_assign_binop():
    node = ast.Assign(
        targets=[ast.Name(id="y", ctx=ast.Store())],
        value=ast.BinOp(
            left=ast.Constant(value=1), op=ast.Add(), right=ast.Constant(value=2)
        ),
    )

    e = AssignEmitter()
    result = e.emit(node)
    assert result.strip() == "local y = 1 + 2"


def test_assign_unsupported_expr():
    node = ast.Assign(
        targets=[ast.Name(id="w", ctx=ast.Store())],
        value=ast.Tuple(elts=[], ctx=ast.Load()),
    )

    e = AssignEmitter()
    with pytest.raises(NotImplementedError):
        e.emit(node)

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


def test_global_mark_var_assignment():
    node = ast.parse("x = Global.mark_var(10)").body[0]

    e = AssignEmitter()
    result = e.emit(node)  # type: ignore[arg-type]
    assert result.strip() == "x = 10"


def test_global_mark_var_requires_value():
    node = ast.parse("x = Global.mark_var()").body[0]

    e = AssignEmitter()
    with pytest.raises(ValueError):
        e.emit(node)  # type: ignore[arg-type]


def test_global_annotation_assignment():
    node = ast.parse("z: Global = 5").body[0]

    e = AssignEmitter()
    result = e.emit(node)  # type: ignore[arg-type]
    assert result.strip() == "z = 5"


def test_annotation_without_global_stays_local():
    node = ast.parse("q: int = 7").body[0]

    e = AssignEmitter()
    result = e.emit(node)  # type: ignore[arg-type]
    assert result.strip() == "local q = 7"


def test_global_annotation_without_value_defaults_to_nil():
    node = ast.parse("p: Global").body[0]

    e = AssignEmitter()
    result = e.emit(node)  # type: ignore[arg-type]
    assert result.strip() == "p = nil"

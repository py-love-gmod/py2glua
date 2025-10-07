import ast

import pytest

from py2glua.convertor.binop_emitter import BinOpEmitter
from py2glua.exceptions import CompileError


def test_simple_add():
    node = ast.BinOp(
        left=ast.Constant(value=1),
        op=ast.Add(),
        right=ast.Constant(value=2),
    )

    e = BinOpEmitter()
    assert e.emit(node) == "1 + 2"


def test_nested_operations():
    node = ast.BinOp(
        left=ast.BinOp(
            left=ast.Constant(value=1),
            op=ast.Add(),
            right=ast.Constant(value=2),
        ),
        op=ast.Mult(),
        right=ast.Constant(value=3),
    )

    e = BinOpEmitter()
    assert e.emit(node) == "1 + 2 * 3"


def test_unsupported_operator():
    node = ast.BinOp(
        left=ast.Constant(value=1),
        op=ast.BitOr(),
        right=ast.Constant(value=2),
    )

    e = BinOpEmitter()
    with pytest.raises(CompileError):
        e.emit(node)


def test_unsupported_expression():
    node = ast.BinOp(
        left=ast.Tuple(elts=[], ctx=ast.Load()),
        op=ast.Add(),
        right=ast.Constant(value=2),
    )

    e = BinOpEmitter()
    with pytest.raises(NotImplementedError):
        e.emit(node)

import ast

import pytest
from source.py2glua.convertor.return_emitter import ReturnEmitter


def test_return_constant():
    node = ast.Return(value=ast.Constant(value=5))
    e = ReturnEmitter()
    result = e.emit(node)
    assert result.strip() == "return 5"


def test_return_binop():
    node = ast.Return(
        value=ast.BinOp(
            left=ast.Name(id="a", ctx=ast.Load()),
            op=ast.Mult(),
            right=ast.Constant(value=10),
        )
    )
    e = ReturnEmitter()
    result = e.emit(node)
    assert result.strip() == "return a * 10"


def test_return_unsupported_expr():
    node = ast.Return(value=ast.List(elts=[], ctx=ast.Load()))
    e = ReturnEmitter()
    with pytest.raises(NotImplementedError):
        e.emit(node)

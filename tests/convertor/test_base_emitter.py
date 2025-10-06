import ast

import pytest

from py2glua.convertor.base_emitter import NodeEmitter


def test_emit_not_implemented():
    emitter = NodeEmitter()
    node = ast.Constant(value=123)
    with pytest.raises(NotImplementedError):
        emitter.emit(node)


def test_indent_levels():
    e = NodeEmitter()
    assert e._indent(0) == ""
    assert e._indent(1) == "    "
    assert e._indent(2) == "        "
    assert e._indent(3) == "            "


def test_default_for_expr_constant():
    e = NodeEmitter()
    node = ast.Constant(value=42)
    result = e._default_for_expr(node)
    assert result == repr(42)


def test_default_for_expr_name():
    e = NodeEmitter()
    node = ast.Name(id="x")
    result = e._default_for_expr(node)
    assert result == "x"


def test_default_for_expr_unsupported():
    e = NodeEmitter()
    node = ast.BinOp(
        left=ast.Constant(value=1),
        op=ast.Add(),
        right=ast.Constant(value=2),
    )

    result = e._default_for_expr(node)
    assert result is None

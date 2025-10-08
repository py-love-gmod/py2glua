import ast

from py2glua.glua import Global


def test_mark_var_allows_primitives():
    assert Global.mark_var(5) == 5


def test_is_func_decorator_detects_attribute():
    node = ast.parse("@Global.func\ndef foo():\n    pass").body[0].decorator_list[0]  # type: ignore
    assert Global.is_func_decorator(node)


def test_unwrap_mark_var_call_returns_inner_value():
    call = ast.parse("Global.mark_var(value=42)").body[0].value  # type: ignore
    is_global, inner = Global.unwrap_mark_var_call(call)
    assert is_global is True
    assert isinstance(inner, ast.Constant)
    assert inner.value == 42

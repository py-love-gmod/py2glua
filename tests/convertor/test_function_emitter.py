import ast

from py2glua.convertor.function_emitter import FunctionEmitter


def _fn_node(src: str) -> ast.FunctionDef:
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            return n

    raise AssertionError("No function found in source")


def test_function_assign_and_return():
    src = """
def add(a, b):
    c = a + b
    return c
"""
    node = _fn_node(src)
    e = FunctionEmitter()
    result = e.emit(node)
    expected = "function add(a, b)\n    local c = a + b\n    return c\nend"
    assert result == expected


def test_function_with_indent_level():
    src = """
def mul(a, b):
    d = a * b
    return d
"""
    node = _fn_node(src)
    e = FunctionEmitter()
    result = e.emit(node, indent=1)
    expected = (
        "    function mul(a, b)\n        local d = a * b\n        return d\n    end"
    )
    assert result == expected


def test_ignores_non_supported_stmts_expr_and_pass():
    src = """
def foo():
    print("x")
    a = 1
    pass
    return a
"""
    node = _fn_node(src)
    e = FunctionEmitter()
    result = e.emit(node)
    expected = "function foo()\n    local a = 1\n    return a\nend"
    assert result == expected
    assert "print" not in result

import ast
from pathlib import Path

import pytest

from py2glua.exceptions import DeliberatelyUnsupportedError
from py2glua.ir.builder import IRBuilder
from py2glua.ir.ir_base import (
    Attribute,
    BinOp,
    BinOpType,
    BoolOp,
    BoolOpType,
    Break,
    Call,
    ClassDef,
    Compare,
    Constant,
    Continue,
    DictLiteral,
    ExceptHandler,
    File,
    For,
    FunctionDef,
    If,
    Import,
    ListLiteral,
    Pass,
    Return,
    Subscript,
    Try,
    TupleLiteral,
    UnaryOp,
    UnaryOpType,
    VarLoad,
    VarStore,
    While,
    With,
)


# helpers
def build_ir(src: str) -> File:
    tree = ast.parse(src)
    ir = IRBuilder.build_ir(tree)
    assert isinstance(ir, File)
    return ir


# UNSUPPORTED / ERRORS
@pytest.mark.parametrize(
    "code",
    [
        "async def foo():\n    pass\n",
        "async for i in it:\n    pass\n",
        "async with ctx:\n    pass\n",
    ],
)
def test_unsupported_async_constructs(code):
    tree = ast.parse(code)
    with pytest.raises(DeliberatelyUnsupportedError):
        IRBuilder.build_ir(tree, path=Path("file.py"))


def test_await_not_supported():
    tree = ast.parse("await bar()")
    with pytest.raises(DeliberatelyUnsupportedError):
        IRBuilder.build_ir(tree)


def test_annassign_without_value_raises():
    with pytest.raises(NotImplementedError):
        build_ir("x: int")


def test_unsupported_op_to_str_raises():
    class Dummy(ast.AST): ...

    with pytest.raises(NotImplementedError):
        IRBuilder._op_to_str(Dummy())


# FILE BASICS
def test_empty_file_builds():
    ir = build_ir("")
    assert isinstance(ir, File)
    assert ir.body == []


def test_all_top_nodes_linked_to_file():
    ir = build_ir("def f():\n    pass\nx = 1")
    for node in ir.body:
        assert node.file is ir


# IMPORTS
def test_import_basic_and_from_import():
    ir = build_ir("import math, sys as system\nfrom os import path, getcwd as gcwd")
    imp1, imp2 = ir.body
    assert isinstance(imp1, Import)
    assert isinstance(imp2, Import)
    assert (
        imp1.module is None
        and imp1.names == ["math", "sys"]
        and imp1.aliases == [None, "system"]
    )
    assert (
        imp2.module == "os"
        and imp2.names == ["path", "getcwd"]
        and imp2.aliases == [None, "gcwd"]
    )


# VARIABLES / ASSIGNMENTS
def test_varstore_constant_and_augassign():
    ir = build_ir("x = 1\ny += z")
    s1, s2 = ir.body
    assert (
        isinstance(s1, VarStore)
        and isinstance(s1.value, Constant)
        and s1.value.value == 1
    )
    assert isinstance(s2, VarStore)
    assert isinstance(s2.value, BinOp) and s2.value.op is BinOpType.ADD
    assert isinstance(s2.value.left, VarLoad) and isinstance(s2.value.right, VarLoad)


def test_annassign_with_value_and_binop():
    ir = build_ir("x: float = a + b")
    s = ir.body[0]
    assert isinstance(s, VarStore) and s.annotation == "float"
    assert isinstance(s.value, BinOp) and s.value.op is BinOpType.ADD


# BINARY OPS
@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1 + 2", BinOpType.ADD),
        ("1 - 2", BinOpType.SUB),
        ("2 * 3", BinOpType.MUL),
        ("5 / 2", BinOpType.DIV),
        ("5 % 2", BinOpType.MOD),
        ("5 | 2", BinOpType.BIT_OR),
        ("5 & 2", BinOpType.BIT_AND),
        ("5 ^ 2", BinOpType.BIT_XOR),
        ("5 << 2", BinOpType.BIT_LSHIFT),
        ("5 >> 2", BinOpType.BIT_RSHIFT),
        ("5 // 2", BinOpType.FLOOR_DIV),
        ("2 ** 3", BinOpType.POW),
    ],
)
def test_binops_all(expr, expected):
    ir = build_ir(f"x = {expr}")
    s = ir.body[0]
    assert isinstance(s, VarStore)
    assert isinstance(s.value, BinOp)
    assert s.value.op is expected


# UNARY OPS
@pytest.mark.parametrize(
    "expr, expected",
    [
        ("+x", UnaryOpType.POS),
        ("-x", UnaryOpType.NEG),
        ("not x", UnaryOpType.NOT),
        ("~x", UnaryOpType.BIT_NOT),
    ],
)
def test_unary_ops_all(expr, expected):
    ir = build_ir(f"x = {expr}")
    s = ir.body[0]
    assert isinstance(s, VarStore)
    assert isinstance(s.value, UnaryOp)
    assert s.value.op is expected


# BOOL / COMPARE
def test_bool_op_precedence_and_or():
    ir = build_ir("x = a and b or c")
    s = ir.body[0]
    assert isinstance(s, VarStore)
    assert isinstance(s.value, BoolOp)
    assert s.value.op is BoolOpType.OR


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("a == b", BoolOpType.EQ),
        ("a != b", BoolOpType.NE),
        ("a < b", BoolOpType.LT),
        ("a <= b", BoolOpType.LE),
        ("a > b", BoolOpType.GT),
        ("a >= b", BoolOpType.GE),
        ("a is b", BoolOpType.IS),
        ("a is not b", BoolOpType.IS_NOT),
        ("a in b", BoolOpType.IN),
        ("a not in b", BoolOpType.NOT_IN),
    ],
)
def test_compare_all(expr, expected):
    ir = build_ir(f"x = {expr}")
    s = ir.body[0]
    assert isinstance(s, VarStore)
    assert isinstance(s.value, Compare)
    assert s.value.op is expected


# COLLECTIONS
def test_list_tuple_dict_and_empty_dict():
    ir = build_ir('a = [1,2,3]\nb = (x,y)\nc = {"foo": 42, "bar": z}\nd = {}')
    a, b, c, d = ir.body
    assert isinstance(a, VarStore)
    assert isinstance(b, VarStore)
    assert isinstance(c, VarStore)
    assert isinstance(d, VarStore)
    assert isinstance(a.value, ListLiteral) and all(
        isinstance(e, Constant) for e in a.value.elements
    )
    assert isinstance(b.value, TupleLiteral) and all(
        isinstance(e, VarLoad) for e in b.value.elements
    )
    assert (
        isinstance(c.value, DictLiteral)
        and len(c.value.keys) == len(c.value.values) == 2
    )
    assert (
        isinstance(d.value, DictLiteral) and d.value.keys == [] and d.value.values == []
    )


# CALLS / ATTRIBUTES / SUBSCRIPT
def test_call_positional_and_keywords():
    ir = build_ir("res = func(1, a=2, flag=True)")
    s = ir.body[0]
    assert isinstance(s, VarStore)
    assert isinstance(s.value, Call)
    call = s.value
    assert len(call.args) == 1 and isinstance(call.args[0], Constant)
    assert hasattr(call, "kwargs")
    assert "a" in call.kwargs and "flag" in call.kwargs
    assert (
        isinstance(call.kwargs["flag"], Constant) and call.kwargs["flag"].value is True
    )


def test_attribute_chain_and_subscript_index():
    ir = build_ir("x = obj.field.subfield[5]")
    s = ir.body[0]
    assert isinstance(s, VarStore)
    assert isinstance(s.value, Subscript)
    sub = s.value
    assert isinstance(sub.index, Constant) and sub.index.value == 5
    assert isinstance(sub.value, Attribute)
    assert isinstance(sub.value.value, Attribute)
    inner = sub.value.value
    assert isinstance(inner.value, VarLoad) and inner.value.name == "obj"


# FUNCTIONS
def test_function_args_body_decorator_and_return_none():
    ir = build_ir("@decor\ndef f(a,b):\n    return")
    fn = ir.body[0]
    assert isinstance(fn, FunctionDef)
    assert fn.decorators == ["decor"]
    assert fn.args == ["a", "b"]
    assert isinstance(fn.body[0], Return) and fn.body[0].value is None


def test_function_nested_ops():
    ir = build_ir("def f():\n    x = not (a + 1 >= b or c)\n    return x")
    fn = ir.body[0]
    assert isinstance(fn, FunctionDef)
    assert any(isinstance(ch, Return) for ch in fn.body)


# CLASSES
def test_class_simple_bases_decorators_and_method():
    ir = build_ir("@dec\nclass A(B,C):\n    x = 1\n    def m(self):\n        return 2")
    cls = ir.body[0]
    assert isinstance(cls, ClassDef)
    assert cls.decorators == ["dec"]
    assert len(cls.bases) == 2
    assert any(isinstance(ch, VarStore) for ch in cls.body)
    methods = [ch for ch in cls.body if isinstance(ch, FunctionDef)]
    assert methods and methods[0].name == "m"


def test_nested_class_and_method_return_const():
    ir = build_ir(
        "class Outer:\n    class Inner:\n        def m(self):\n            return 1"
    )
    outer = ir.body[0]
    assert isinstance(outer, ClassDef)
    inner = outer.body[0]
    assert isinstance(inner, ClassDef)
    fn = inner.body[0]
    assert isinstance(fn, FunctionDef)
    assert isinstance(fn.body[0], Return)
    assert isinstance(fn.body[0].value, Constant) and fn.body[0].value.value == 1


# CONTROL FLOW
def test_if_elif_else_and_compare():
    ir = build_ir("if x < 5: a=1\nelif x < 10: a=2\nelse: a=3")
    node = ir.body[0]
    assert isinstance(node, If)
    assert isinstance(node.test, Compare)
    assert node.body and node.orelse


def test_while_with_break_and_for_with_continue():
    ir = build_ir("while True:\n    break\nfor i in range(3):\n    continue")
    nodes = list(ir.walk())
    assert any(isinstance(n, Break) for n in nodes)
    assert any(isinstance(n, Continue) for n in nodes)


def test_with_context_and_target():
    ir = build_ir("with ctx as c:\n    x = 1")
    w = ir.body[0]
    assert isinstance(w, With)
    assert isinstance(w.context, VarLoad)
    assert isinstance(w.target, VarLoad) and w.target.name == "c"
    assert w.body and isinstance(w.body[0], VarStore)


def test_try_handlers_else_finally_and_multiple_handlers():
    ir1 = build_ir(
        "try:\n    a=1\nexcept ValueError as e:\n    a=2\nelse:\n    a=3\nfinally:\n    a=4"
    )
    t = ir1.body[0]
    assert isinstance(t, Try) and len(t.handlers) == 1 and t.orelse and t.finalbody

    ir2 = build_ir(
        "try:\n    a=1\nexcept KeyError:\n    a=2\nexcept Exception as e:\n    a=3"
    )
    tt = ir2.body[0]
    assert isinstance(tt, Try) and len(tt.handlers) == 2
    assert all(isinstance(h, ExceptHandler) for h in tt.handlers)


def test_while_basic_with_compare_and_body_structure():
    src = """
while x < 10:
    x += 1
"""
    ir = build_ir(src)
    assert len(ir.body) == 1
    loop = ir.body[0]
    assert isinstance(loop, While)
    assert isinstance(loop.test, Compare)
    assert loop.test.op is BoolOpType.LT
    assert isinstance(loop.test.left, VarLoad)
    assert isinstance(loop.test.right, Constant)
    assert len(loop.body) == 1
    stmt = loop.body[0]
    assert isinstance(stmt, VarStore)
    assert isinstance(stmt.value, BinOp)
    assert stmt.value.op is BinOpType.ADD
    assert stmt.parent is loop
    assert loop.file is ir


def test_while_with_else_and_nested_pass():
    src = "while flag:\n    if ready:\n        pass\nelse:\n    result = 42\n"
    ir = build_ir(src)
    loop = ir.body[0]
    assert isinstance(loop, While)
    assert isinstance(loop.test, VarLoad)
    inner_if = loop.body[0]
    assert isinstance(inner_if, If)
    assert isinstance(inner_if.body[0], Pass)
    assert isinstance(loop.orelse[0], VarStore)
    assert isinstance(loop.orelse[0].value, Constant)


def test_pass_as_top_level_statement_and_in_class_function():
    src = """
pass

class Foo:
    pass

def bar():
    pass
"""
    ir = build_ir(src)
    top = ir.body[0]
    assert isinstance(top, Pass)
    assert top.parent is ir
    cls = ir.body[1]
    assert isinstance(cls, ClassDef)
    assert len(cls.body) == 1
    assert isinstance(cls.body[0], Pass)
    fn = ir.body[2]
    assert isinstance(fn, FunctionDef)
    assert isinstance(fn.body[0], Pass)
    assert fn.body[0].parent is fn


def test_for_basic_loop():
    src = "for i in range(3):\n    x = i"
    ir = build_ir(src)
    node = ir.body[0]
    assert isinstance(node, For)
    assert isinstance(node.target, VarLoad)
    assert node.target.name == "i"
    assert isinstance(node.iter, Call)
    assert isinstance(node.iter.func, VarLoad)
    assert node.iter.func.name == "range"
    assert isinstance(node.body[0], VarStore)
    assert node.orelse == []


def test_for_with_else_clause():
    src = "for i in range(5):\n    do(i)\nelse:\n    finished = True\n"
    ir = build_ir(src)
    loop = ir.body[0]
    assert isinstance(loop, For)
    assert len(loop.body) == 1
    assert len(loop.orelse) == 1
    store = loop.orelse[0]
    assert isinstance(store, VarStore)
    assert store.name == "finished"
    assert isinstance(store.value, Constant)
    assert store.value.value is True


def test_for_nested_loop():
    src = "for x in range(2):\n    for y in range(3):\n        print(x, y)\n"
    ir = build_ir(src)
    outer = ir.body[0]
    assert isinstance(outer, For)
    inner = outer.body[0]
    assert isinstance(inner, For)
    assert isinstance(inner.iter, Call)
    assert isinstance(inner.iter.func, VarLoad)
    assert inner.iter.func.name == "range"
    assert isinstance(inner.body[0], Call)
    assert isinstance(inner.body[0].func, VarLoad)
    assert inner.body[0].func.name == "print"


def test_for_with_break_and_continue():
    src = (
        "for i in range(10):\n"
        "    if i == 5:\n"
        "        break\n"
        "    if i % 2 == 0:\n"
        "        continue\n"
    )
    ir = build_ir(src)
    loop = ir.body[0]
    assert isinstance(loop, For)
    assert any(isinstance(ch, Break) for ch in loop.walk())
    assert any(isinstance(ch, Continue) for ch in loop.walk())


def test_for_with_empty_body():
    src = "for i in []:\n    pass"
    ir = build_ir(src)
    loop = ir.body[0]
    assert isinstance(loop, For)
    assert len(loop.body) == 1
    assert isinstance(loop.body[0], Pass)


# WALK COVERAGE
def test_walk_covers_nested_nodes():
    ir = build_ir("def f():\n    a=[1,2]\n    b={'x':3}\n    return a")
    nodes = list(ir.walk())
    assert any(isinstance(n, ListLiteral) for n in nodes)
    assert any(isinstance(n, DictLiteral) for n in nodes)
    assert any(isinstance(n, Return) for n in nodes)

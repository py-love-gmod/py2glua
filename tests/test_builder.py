import ast

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
    File,
    For,
    FunctionDef,
    If,
    Import,
    ListLiteral,
    Pass,
    Return,
    Subscript,
    TupleLiteral,
    UnaryOp,
    UnaryOpType,
    VarLoad,
    VarStore,
    While,
    With,
)


# region helpers
def build_ir(src: str) -> File:
    tree = ast.parse(src)
    ir = IRBuilder.build_ir(tree)
    assert isinstance(ir, File)
    return ir


# endregion


# region ETC
def test_empty_file():
    ir = build_ir("")
    assert isinstance(ir, File)
    assert ir.body == []


# endregion


# region IMPORTS
def test_import_basic():
    ir = build_ir("import math, sys as system")
    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, Import)
    assert node.names == ["math", "sys"]
    assert node.aliases == [None, "system"]


def test_import_from_basic():
    ir = build_ir("from os import path, system as sys")
    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, Import)
    assert node.module == "os"
    assert node.names == ["path", "system"]
    assert node.aliases == [None, "sys"]


# endregion


# region VARIABLES / CONSTANTS / ASSIGNMENTS
def test_varstore_constant():
    ir = build_ir("x = 1")
    node = ir.body[0]
    assert isinstance(node, VarStore)
    assert node.name == "x"
    assert isinstance(node.value, Constant)
    assert node.value.value == 1


def test_augassign_basic():
    ir = build_ir("x += y")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    value = store.value
    assert isinstance(value, BinOp)
    assert value.op is BinOpType.ADD
    assert isinstance(value.left, VarLoad)
    assert isinstance(value.right, VarLoad)


def test_annassign_constant():
    ir = build_ir("x: int = 10")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    assert store.annotation == "int"
    assert isinstance(store.value, Constant)
    assert store.value.value == 10


def test_annassign_binop():
    ir = build_ir("y: float = a + b")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    assert store.annotation == "float"
    assert isinstance(store.value, BinOp)


# endregion


# region BINARY / UNARY / BOOL / COMPARE
def test_binop_basic():
    ir = build_ir("y = 2 * 3")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    assert isinstance(store.value, BinOp)
    assert store.value.op is BinOpType.MUL


def test_nested_binop():
    ir = build_ir("z = a + b * c")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    outer = store.value
    assert isinstance(outer, BinOp)
    assert outer.op is BinOpType.ADD
    assert isinstance(outer.right, BinOp)
    assert outer.right.op is BinOpType.MUL


def test_unary_not():
    ir = build_ir("x = not y")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    op = store.value
    assert isinstance(op, UnaryOp)
    assert op.op is UnaryOpType.NOT


def test_unary_neg():
    ir = build_ir("x = -y")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    op = store.value
    assert isinstance(op, UnaryOp)
    assert op.op is UnaryOpType.NEG


def test_bool_and_or():
    ir = build_ir("x = a and b or c")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    top = store.value
    assert isinstance(top, BoolOp)
    assert top.op is BoolOpType.OR


def test_compare_eq():
    ir = build_ir("x = a == b")
    body = ir.body[0]
    assert isinstance(body, VarStore)
    cmp = body.value
    assert isinstance(cmp, Compare)
    assert cmp.op is BoolOpType.EQ


def test_compare_ge():
    ir = build_ir("x = a >= b")
    body = ir.body[0]
    assert isinstance(body, VarStore)
    cmp = body.value
    assert isinstance(cmp, Compare)
    assert cmp.op is BoolOpType.GE


# endregion


# region COLLECTIONS
def test_list_literal():
    ir = build_ir("a = [1, 2, 3]")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    assert isinstance(store.value, ListLiteral)
    assert all(isinstance(e, Constant) for e in store.value.elements)


def test_tuple_literal():
    ir = build_ir("b = (x, y)")
    store = ir.body[0]
    assert isinstance(store, VarStore)
    assert isinstance(store.value, TupleLiteral)
    assert all(isinstance(e, VarLoad) for e in store.value.elements)


def test_dict_literal():
    ir = build_ir('c = {"foo": 42, "bar": z}')
    store = ir.body[0]
    assert isinstance(store, VarStore)
    assert isinstance(store.value, DictLiteral)
    assert len(store.value.keys) == len(store.value.values) == 2


# endregion


# region CALLS / ATTRIBUTES / SUBSCRIPTS
def test_call_simple():
    ir = build_ir("print(x)")
    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, Call)
    func = node.func
    assert isinstance(func, VarLoad)
    assert func.name == "print"
    assert len(node.args) == 1
    arg = node.args[0]
    assert isinstance(arg, VarLoad)
    assert arg.name == "x"


def test_attribute_access():
    ir = build_ir("x = player.GetPos")
    assert len(ir.body) == 1
    store = ir.body[0]
    assert isinstance(store, VarStore)
    value = store.value
    assert isinstance(value, Attribute)
    val = value.value
    assert isinstance(val, VarLoad)
    assert val.name == "player"
    assert value.attr == "GetPos"


def test_subscript_access():
    ir = build_ir("x = arr[5]")
    assert isinstance(ir.body[0], VarStore)
    sub = ir.body[0].value
    assert isinstance(sub, Subscript)
    assert isinstance(sub.index, Constant)
    assert sub.index.value == 5


# endregion


# region FUNCTIONS
def test_function_simple():
    src = """
def foo(a, b):
    x = a + b
    return x
"""
    ir = build_ir(src)
    fn = ir.body[0]
    assert isinstance(fn, FunctionDef)
    assert fn.name == "foo"
    assert fn.args == ["a", "b"]
    assert any(isinstance(ch, Return) for ch in fn.body)


def test_function_with_decorator():
    src = """
@decorator
def greet(name):
    return "Hello " + name
"""
    ir = build_ir(src)
    fn = ir.body[0]
    assert isinstance(fn, FunctionDef)
    assert fn.decorators == ["decorator"]


def test_return_none():
    src = """
def f():
    return
"""
    ir = build_ir(src)
    fn = ir.body[0]
    assert isinstance(fn, FunctionDef)
    assert isinstance(fn.body[0], Return)
    assert fn.body[0].value is None


def test_pass_statement():
    src = """
def foo():
    pass
"""
    ir = build_ir(src)
    fn = ir.body[0]
    assert isinstance(fn, FunctionDef)
    assert isinstance(fn.body[0], Pass)


# endregion


# region CONTROL FLOW
def test_if_basic():
    src = """
if a > b:
    x = 1
else:
    x = 2
"""
    ir = build_ir(src)
    node = ir.body[0]
    assert isinstance(node, If)
    assert isinstance(node.test, Compare)


def test_while_basic():
    src = """
while x < 10:
    x += 1
"""
    ir = build_ir(src)
    loop = ir.body[0]
    assert isinstance(loop, While)
    assert isinstance(loop.test, Compare)
    assert any(isinstance(ch, VarStore) for ch in loop.body)


def test_for_basic():
    src = """
for i in range(3):
    print(i)
"""
    ir = build_ir(src)
    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, For)
    loop = node
    assert isinstance(loop.target, VarLoad)
    target = loop.target
    assert target.name == "i"
    assert isinstance(loop.iter, Call)
    iter_call = loop.iter
    assert isinstance(iter_call.func, VarLoad)
    assert iter_call.func.name == "range"
    assert all(isinstance(ch, Call) for ch in loop.body)


def test_with_basic():
    src = """
with ctx:
    x = 1
"""
    ir = build_ir(src)
    node = ir.body[0]
    assert isinstance(node, With)
    assert isinstance(node.context, VarLoad)
    assert node.body and isinstance(node.body[0], VarStore)


def test_with_as():
    src = """
with manager as m:
    y = 2
"""
    ir = build_ir(src)
    w = ir.body[0]
    assert isinstance(w, With)
    assert isinstance(w.target, VarLoad)
    assert w.target.name == "m"


def test_break():
    src = """
while True:
    break
"""
    ir = build_ir(src)
    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, While)
    loop = node
    assert any(isinstance(ch, Break) for ch in loop.body)


def test_continue():
    src = """
while True:
    continue
"""
    ir = build_ir(src)
    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, While)
    loop = node
    assert any(isinstance(ch, Continue) for ch in loop.body)


# endregion


# region CLASSES
def test_class_simple():
    src = """
class Foo:
    pass
"""
    ir = build_ir(src)
    cls = ir.body[0]
    assert isinstance(cls, ClassDef)
    assert cls.name == "Foo"
    assert len(cls.body) == 1
    assert isinstance(cls.body[0], Pass)


def test_class_with_base():
    src = """
class Bar(Base):
    pass
"""
    ir = build_ir(src)
    cls = ir.body[0]
    assert isinstance(cls, ClassDef)
    assert len(cls.bases) == 1
    base = cls.bases[0]
    assert isinstance(base, VarLoad)
    assert base.name == "Base"


def test_class_with_method():
    src = """
class C:
    def m(self):
        return 1
"""
    ir = build_ir(src)
    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, ClassDef)
    cls = node
    assert len(cls.body) == 1
    method = cls.body[0]
    assert isinstance(method, FunctionDef)
    fn = method
    assert fn.name == "m"
    assert fn.args == ["self"]
    assert len(fn.body) == 1
    stmt = fn.body[0]
    assert isinstance(stmt, Return)
    if isinstance(stmt.value, Constant):
        assert stmt.value.value == 1


def test_class_with_multiple_bases_and_decorator():
    src = """
@decor
class Baz(A, B):
    x = 10
"""
    ir = build_ir(src)
    cls = ir.body[0]
    assert isinstance(cls, ClassDef)
    assert cls.decorators == ["decor"]
    assert [b.name for b in cls.bases if isinstance(b, VarLoad)] == ["A", "B"]
    assert isinstance(cls.body[0], VarStore)


# endregion

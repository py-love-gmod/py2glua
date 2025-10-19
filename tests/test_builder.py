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
    Compare,
    Constant,
    Continue,
    DictLiteral,
    File,
    For,
    Function,
    If,
    ListLiteral,
    Return,
    Subscript,
    TupleLiteral,
    UnaryOp,
    UnaryOpType,
    VarLoad,
    VarStore,
    While,
)


def build_ir(src: str) -> File:
    tree = ast.parse(src)
    ir = IRBuilder.build_ir(tree)
    assert isinstance(ir, File)
    return ir


def test_simple_assign() -> None:
    src = "x = 1"
    ir = build_ir(src)
    assert len(ir.body) == 1

    node = ir.body[0]
    assert isinstance(node, VarStore)
    assert node.name == "x"

    value = node.value
    assert isinstance(value, Constant)
    assert value.value == 1
    assert value.parent is node


def test_binary_operation() -> None:
    src = "y = 2 * 3"
    ir = build_ir(src)
    assert len(ir.body) == 1

    store = ir.body[0]
    assert isinstance(store, VarStore)
    assert store.name == "y"

    binop = store.value
    assert isinstance(binop, BinOp)
    assert binop.op is BinOpType.MUL

    left, right = binop.left, binop.right
    assert isinstance(left, Constant)
    assert isinstance(right, Constant)
    assert left.value == 2
    assert right.value == 3


def test_augassign() -> None:
    src = "x += y"
    ir = build_ir(src)
    store = ir.body[0]
    assert isinstance(store, VarStore)
    assert store.name == "x"

    value = store.value
    assert isinstance(value, BinOp)
    assert value.op is BinOpType.ADD

    left, right = value.left, value.right
    assert isinstance(left, VarLoad)
    assert isinstance(right, VarLoad)
    assert left.name == "x"
    assert right.name == "y"


def test_function_def() -> None:
    src = """
def foo(a, b):
    x = a + b
    return x
"""
    ir = build_ir(src)
    assert len(ir.body) == 1

    fn = ir.body[0]
    assert isinstance(fn, Function)
    assert fn.name == "foo"
    assert fn.args == ["a", "b"]

    assert len(fn.body) == 2
    store, ret = fn.body
    assert isinstance(store, VarStore)
    assert isinstance(ret, Return)

    binop = store.value
    assert isinstance(binop, BinOp)
    assert binop.op is BinOpType.ADD

    value = ret.value
    assert isinstance(value, VarLoad)
    assert value.name == "x"


def test_nested_binop() -> None:
    src = "z = a + b * c"
    ir = build_ir(src)
    store = ir.body[0]
    assert isinstance(store, VarStore)

    outer = store.value
    assert isinstance(outer, BinOp)
    assert outer.op is BinOpType.ADD

    right = outer.right
    assert isinstance(right, BinOp)
    assert right.op is BinOpType.MUL
    assert isinstance(right.left, VarLoad)
    assert isinstance(right.right, VarLoad)

    # Род связи
    assert right.parent is outer
    assert outer.parent is store
    assert store.parent is ir


def test_return_none() -> None:
    src = """
def f():
    return
"""
    ir = build_ir(src)
    fn = ir.body[0]
    assert isinstance(fn, Function)
    assert len(fn.body) == 1

    ret = fn.body[0]
    assert isinstance(ret, Return)
    assert ret.value is None


def test_bool_and_or() -> None:
    src = "x = a and b or c"
    ir = build_ir(src)
    store = ir.body[0]
    assert isinstance(store, VarStore)

    top = store.value
    assert isinstance(top, BoolOp)
    assert top.op is BoolOpType.OR

    left = top.left
    assert isinstance(left, BoolOp)
    assert left.op is BoolOpType.AND

    assert isinstance(left.left, VarLoad)
    assert left.left.name == "a"
    assert isinstance(left.right, VarLoad)
    assert left.right.name == "b"
    assert isinstance(top.right, VarLoad)
    assert top.right.name == "c"


def test_not_unary() -> None:
    src = "x = not y"
    ir = build_ir(src)
    store = ir.body[0]
    assert isinstance(store, VarStore)

    op = store.value
    assert isinstance(op, UnaryOp)
    assert op.op is UnaryOpType.NOT

    operand = op.operand
    assert isinstance(operand, VarLoad)
    assert operand.name == "y"


def test_compare_eq() -> None:
    src = "x = a == b"
    ir = build_ir(src)
    store = ir.body[0]
    assert isinstance(store, VarStore)

    cmp = store.value
    assert isinstance(cmp, Compare)
    assert cmp.op is BoolOpType.EQ

    left, right = cmp.left, cmp.right
    assert isinstance(left, VarLoad)
    assert isinstance(right, VarLoad)
    assert left.name == "a"
    assert right.name == "b"


def test_annassign_simple() -> None:
    src = "x: int = 10"
    ir = build_ir(src)
    store = ir.body[0]

    assert isinstance(store, VarStore)
    assert store.name == "x"
    assert isinstance(store.value, Constant)
    assert store.value.value == 10
    assert store.value.parent is store

    if hasattr(store, "annotation"):
        assert store.annotation == "int"


def test_annassign_binop() -> None:
    src = "y: float = a + b"
    ir = build_ir(src)
    store = ir.body[0]

    assert isinstance(store, VarStore)
    assert store.name == "y"

    value = store.value
    assert isinstance(value, BinOp)
    assert value.op is BinOpType.ADD
    assert isinstance(value.left, VarLoad)
    assert isinstance(value.right, VarLoad)

    assert value.left.name == "a"
    assert value.right.name == "b"
    assert value.parent is store

    if hasattr(store, "annotation"):
        assert store.annotation == "float"


def test_call_simple():
    src = "print(x)"
    ir = build_ir(src)
    node = ir.body[0]
    assert isinstance(node, Call)
    assert isinstance(node.func, VarLoad)
    assert node.func.name == "print"
    assert len(node.args) == 1
    assert isinstance(node.args[0], VarLoad)
    assert node.args[0].name == "x"
    assert node.args[0].parent is node


def test_if_simple():
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
    assert len(node.body) == 1
    assert len(node.orelse) == 1
    assert isinstance(node.body[0], VarStore)
    assert isinstance(node.orelse[0], VarStore)


def test_while_loop():
    src = """
while x < 10:
    x += 1
"""
    ir = build_ir(src)
    loop = ir.body[0]
    assert isinstance(loop, While)
    assert isinstance(loop.test, Compare)
    assert len(loop.body) == 1
    assert isinstance(loop.body[0], VarStore)


def test_for_loop():
    src = """
for i in range(3):
    print(i)
"""
    ir = build_ir(src)
    loop = ir.body[0]
    assert isinstance(loop, For)
    assert isinstance(loop.target, VarLoad)
    assert isinstance(loop.iter, Call)
    assert loop.target.name == "i"


def test_break_continue():
    src = """
while True:
    break
    continue
"""
    ir = build_ir(src)
    loop = ir.body[0]
    assert isinstance(loop, While)
    assert any(isinstance(ch, Break) for ch in loop.body)
    assert any(isinstance(ch, Continue) for ch in loop.body)


def test_attribute():
    src = "x = player.GetPos"
    ir = build_ir(src)
    store = ir.body[0]
    assert isinstance(store, VarStore)
    attr = store.value
    assert isinstance(attr, Attribute)
    assert isinstance(attr.value, VarLoad)
    assert attr.value.name == "player"
    assert attr.attr == "GetPos"


def test_subscript():
    src = "x = arr[5]"
    ir = build_ir(src)

    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, VarStore)
    sub = node.value

    assert isinstance(sub, Subscript)
    assert isinstance(sub.value, VarLoad)
    assert isinstance(sub.index, Constant)
    assert sub.index.value == 5


def test_unary_ops():
    src = "x = -y"
    ir = build_ir(src)

    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, VarStore)
    op = node.value

    assert isinstance(op, UnaryOp)
    assert op.op is UnaryOpType.NEG
    assert isinstance(op.operand, VarLoad)
    assert op.operand.name == "y"


def test_compare_ops():
    src = "x = a >= b"
    ir = build_ir(src)

    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, VarStore)
    cmp = node.value

    assert isinstance(cmp, Compare)
    assert cmp.op is BoolOpType.GE
    assert isinstance(cmp.left, VarLoad)
    assert isinstance(cmp.right, VarLoad)


def test_list_tuple_dict():
    src = """
a = [1, 2, 3]
b = (x, y)
c = {"foo": 42, "bar": z}
"""
    ir = build_ir(src)
    assert len(ir.body) == 3

    lst = ir.body[0]
    assert isinstance(lst, VarStore)
    assert isinstance(lst.value, ListLiteral)
    assert all(isinstance(e, Constant) for e in lst.value.elements)

    tup = ir.body[1]
    assert isinstance(tup, VarStore)
    assert isinstance(tup.value, TupleLiteral)
    assert all(isinstance(e, VarLoad) for e in tup.value.elements)

    dct = ir.body[2]
    assert isinstance(dct, VarStore)
    assert isinstance(dct.value, DictLiteral)
    assert len(dct.value.keys) == len(dct.value.values) == 2
    assert all(
        isinstance(k, Constant) or isinstance(k, VarLoad)
        for k in dct.value.keys + dct.value.values
    )

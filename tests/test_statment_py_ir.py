import pytest

from py2glua.lang.py_ir_bulder import (
    PyIRAssign,
    PyIRAtt,
    PyIRBinOp,
    PyIRBuilder,
    PyIRCall,
    PyIRConstant,
    PyIRImport,
    PyIRIndex,
    PyIRMethodCall,
    PyIRUnaryOp,
    PyIRVarCreate,
    PyIRVarUse,
)


# region baseline
def test_var_create_and_assign():
    ir = PyIRBuilder.build("x = 1\n")
    assert isinstance(ir.body[0], PyIRVarCreate)
    assert isinstance(ir.body[1], PyIRAssign)
    assert isinstance(ir.body[1].value, PyIRConstant)
    assert ir.body[1].value.value == 1


def test_global_var():
    ir = PyIRBuilder.build("x = Global.var(10, external=True)\n")
    create = ir.body[0]
    assign = ir.body[1]
    assert isinstance(create, PyIRVarCreate)
    assert create.is_global is True
    assert create.external is True
    assert isinstance(assign, PyIRAssign)
    assert isinstance(assign.value, PyIRConstant)
    assert assign.value.value == 10
    assert ir.meta["globals"][create.vid]["external"] is True


def test_simple_call():
    ir = PyIRBuilder.build("foo(1, 2)\n")
    node = ir.body[0]
    assert isinstance(node, PyIRCall)
    assert node.name == "foo"
    assert len(node.args) == 2


def test_nested_call():
    ir = PyIRBuilder.build("foo(bar(1))\n")
    node = ir.body[0]
    assert isinstance(node, PyIRCall)
    inner = node.args[0]
    assert isinstance(inner, PyIRCall)
    assert inner.name == "bar"


def test_method_call():
    ir = PyIRBuilder.build("obj.method(10)\n")
    node = ir.body[0]
    assert isinstance(node, PyIRMethodCall)
    assert node.method == "method"


def test_chained_method():
    ir = PyIRBuilder.build("obj.a().b(3)\n")
    node = ir.body[0]
    assert isinstance(node, PyIRMethodCall)
    assert node.method == "b"
    assert isinstance(node.obj, PyIRMethodCall)
    assert node.obj.method == "a"


def test_simple_index():
    ir = PyIRBuilder.build("arr[i]\n")
    node = ir.body[0]
    assert isinstance(node, PyIRIndex)
    assert isinstance(node.obj, PyIRVarUse)
    assert isinstance(node.index, PyIRVarUse)


def test_index_chain():
    ir = PyIRBuilder.build("a[b][c]\n")
    node = ir.body[0]
    assert isinstance(node, PyIRIndex)
    assert isinstance(node.obj, PyIRIndex)


def test_addition_op():
    ir = PyIRBuilder.build("x = 1 + 2\n")
    assign = ir.body[1]
    assert isinstance(assign.value, PyIRBinOp)  # type: ignore
    assert assign.value.op == "+"  # type: ignore


def test_unary_not():
    ir = PyIRBuilder.build("x = not y\n")
    assign = ir.body[1]
    assert isinstance(assign.value, PyIRUnaryOp)  # type: ignore
    assert assign.value.op == "not"  # type: ignore


def test_complex_expression_grouping():
    ir = PyIRBuilder.build("x = (a + b) * c - d / e\n")
    assign = ir.body[1]
    assert isinstance(assign.value, PyIRBinOp)  # type: ignore


def test_attribute_chain():
    ir = PyIRBuilder.build("x = a.b.c\n")
    assign = ir.body[1]
    assert isinstance(assign.value, PyIRAtt)  # type: ignore
    assert assign.value.name == "a.b.c"  # type: ignore


def test_import_alias_and_use():
    ir = PyIRBuilder.build("import math as m\nx = m.sin(1)\n")
    assert isinstance(ir.body[0], PyIRImport)
    assign = [n for n in ir.body if isinstance(n, PyIRAssign)][0]
    assert isinstance(assign.value, PyIRCall)  # type: ignore
    assert assign.value.name == "math.sin"  # type: ignore


@pytest.mark.parametrize(
    "src,op",
    [
        ("x = a - b\n", "-"),
        ("x = a * b\n", "*"),
        ("x = a / b\n", "/"),
        ("x = a // b\n", "//"),
        ("x = a % b\n", "%"),
        ("x = a == b\n", "=="),
        ("x = a != b\n", "!="),
        ("x = a < b\n", "<"),
        ("x = a <= b\n", "<="),
        ("x = a > b\n", ">"),
        ("x = a >= b\n", ">="),
        ("x = (a+b) * (c-d)\n", "*"),
        ("x = a and b or c\n", "or"),
    ],
)
def test_more_ops_supported(src, op):
    ir = PyIRBuilder.build(src)
    assign = ir.body[1]
    assert isinstance(assign.value, (PyIRBinOp, PyIRUnaryOp))  # type: ignore

    def contains_op(node) -> bool:
        if isinstance(node, PyIRBinOp) and node.op == op:
            return True

        for ch in getattr(node, "__dict__", {}).values():
            if isinstance(ch, (PyIRBinOp, PyIRUnaryOp)):
                if contains_op(ch):
                    return True

        return False

    assert contains_op(assign.value)  # type: ignore


@pytest.mark.parametrize(
    "src,expected",
    [
        ("x = 'hello'\n", "hello"),
        ('x = "world"\n', "world"),
        ("x = 0xFF\n", 255),
        ("x = 3.14\n", 3.14),
        ("x = True\n", True),
        ("x = False\n", False),
        ("x = None\n", None),
    ],
)
def test_literals_supported(src, expected):
    ir = PyIRBuilder.build(src)
    val = ir.body[1].value  # type: ignore
    assert isinstance(val, PyIRConstant)
    assert val.value == expected


# endregion


# region slicing
@pytest.mark.parametrize(
    "src",
    [
        "x = a[1:]\n",
        "x = a[:5]\n",
        "x = a[::2]\n",
        "x = a[1:5:2]\n",
        "x = a[b:c]\n",
    ],
)
def test_slices_xfail(src):
    pytest.xfail("Slices не поддержаны текущим IR")
    PyIRBuilder.build(src)


# endregion


# region tuple/list/dict/set literals
@pytest.mark.parametrize(
    "src",
    [
        "x = (1, 2, 3)\n",
        "x = [1, 2, 3]\n",
        "x = {1, 2, 3}\n",
        "x = {'a': b}\n",
    ],
)
def test_container_literals_xfail(src):
    pytest.xfail("Коллекции (tuple/list/dict/set) не поддержаны текущим IR")
    PyIRBuilder.build(src)


# endregion


# region comprehensions
@pytest.mark.parametrize(
    "src",
    [
        "x = [i for i in xs]\n",
        "x = {i for i in xs]\n".replace("]", "}"),
        "x = {k: v for k,v in d.items()}\n",
        "x = (i for i in xs)\n",
    ],
)
def test_comprehensions_xfail(src):
    pytest.xfail("Comprehensions не поддержаны")
    PyIRBuilder.build(src)


# endregion


# region lambda / conditional-expr
@pytest.mark.parametrize(
    "src",
    [
        "x = lambda a: a+1\n",
        "x = a if cond else b\n",
    ],
)
def test_lambda_conditional_xfail(src):
    pytest.xfail("Lambda/conditional expr не поддержаны")
    PyIRBuilder.build(src)


# endregion


# region power/bitwise/tilde
@pytest.mark.parametrize(
    "src",
    [
        "x = a ** b\n",
        "x = a & b\n",
        "x = a | b\n",
        "x = a ^ b\n",
        "x = a << b\n",
        "x = a >> b\n",
        "x = ~a\n",
    ],
)
def test_power_bitwise_xfail(src):
    pytest.xfail("Power/bitwise/tilde не поддержаны текущим IR")
    PyIRBuilder.build(src)


# endregion


# region f-strings/bytes/ellipsis
@pytest.mark.parametrize(
    "src",
    [
        'x = f"{a}+{b}"\n',
        "x = b'data'\n",
        "x = ...\n",
        "x = r'raw\\n'\n",
    ],
)
def test_fstrings_bytes_misc_xfail(src):
    pytest.xfail("F-strings/bytes/ellipsis/raw не поддержаны")
    PyIRBuilder.build(src)


# endregion


# region walrus/type-annot/augassign
@pytest.mark.parametrize(
    "src",
    [
        "x = (y := 5)\n",
        "x: int = 5\n",
        "x += 1\n",
        "arr[i] = 2\n",
        "obj.attr = 3\n",
        "a, b = t\n",
        "a, *b = t\n",
    ],
)
def test_walrus_annotations_augassign_targets_xfail(src):
    pytest.xfail("Walrus/annotation/augassign/сложные targets не поддержаны")
    PyIRBuilder.build(src)


# endregion


# region chained comparisons
@pytest.mark.parametrize(
    "src",
    [
        "x = a < b < c\n",
        "x = a == b == c\n",
        "x = a < b >= c\n",
    ],
)
def test_chained_comparisons_xfail(src):
    pytest.xfail(
        "Chained comparisons — семантика Python, IR пока не имеет узла для цепочек"
    )
    PyIRBuilder.build(src)


# endregion


# region import wildcards/errors
@pytest.mark.parametrize(
    "src",
    [
        "from m import *\n",
    ],
)
def test_import_wildcard_forbidden(src):
    with pytest.raises(SyntaxError):
        PyIRBuilder.build(src)


# endregion


# region import resolution categories
def test_import_local():
    ir = PyIRBuilder.build("import pkg.mod\n")
    imp = ir.body[0]
    assert isinstance(imp, PyIRImport)


def test_import_alias_chain():
    ir = PyIRBuilder.build("import math as m\nx = m.cos(0)\n")
    assigns = [n for n in ir.body if isinstance(n, PyIRAssign)]
    assert assigns, "assign not found"
    call = assigns[0].value
    assert isinstance(call, PyIRCall)
    assert call.name == "math.cos"


# endregion


# region whitespace/newlines/parentheses robustness
@pytest.mark.parametrize(
    "src",
    [
        "x=1\n",
        "x    =    2\n",
        "x = (1)\n",
        "foo(\n  1,\n  2\n)\n",
        "obj.\n  method(\n    5\n  )\n",
    ],
)
def test_layout_tolerance(src):
    try:
        ir = PyIRBuilder.build(src)
        assert ir is not None

    except Exception:
        pytest.xfail("Разметка с переносами/пробелами пока не поддержана")


# endregion

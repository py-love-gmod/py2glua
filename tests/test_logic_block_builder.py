from pathlib import Path

import pytest

from py2glua.lang.py_logic_block_builder import (
    PublicLogicKind,
    PublicLogicNode,
    PyLogicBlockBuilder,
)


# region Helpers
def _dump_kinds(nodes):
    out = []
    for n in nodes:
        out.append(n.kind)
        out.extend(_dump_kinds(n.children))

    return out


def _flatten_kinds(nodes):
    out = []
    for n in nodes:
        out.append(n.kind)
        out.extend(_flatten_kinds(n.children))

    return out


def _kinds_sequence(nodes):
    return [n.kind for n in nodes]


def _has_path(nodes, path):
    if not path:
        return True

    want = path[0]
    for n in nodes:
        if n.kind == want:
            if len(path) == 1:
                return True

            if _has_path(n.children, path[1:]):
                return True

        if _has_path(n.children, path):
            return True

    return False


# endregion


@pytest.mark.parametrize(
    "src,expected_kind",
    [
        ("def f():\n    pass\n", PublicLogicKind.FUNCTION),
        ("class C:\n    pass\n", PublicLogicKind.CLASS),
        ("if True:\n    pass\n", PublicLogicKind.BRANCH),
        ("while True:\n    pass\n", PublicLogicKind.LOOP),
        ("for i in range(1):\n    pass\n", PublicLogicKind.LOOP),
        ("try:\n    pass\nexcept Exception:\n    pass\n", PublicLogicKind.TRY),
        ("with open('a') as f:\n    pass\n", PublicLogicKind.WITH),
    ],
)
def test_basic_block_detection(
    tmp_path: Path, src: str, expected_kind: PublicLogicKind
):
    f = tmp_path / "file.py"
    f.write_text(src, encoding="utf-8-sig")

    result = PyLogicBlockBuilder.build(f)
    kinds = _dump_kinds(result)
    assert expected_kind in kinds


def test_nested_if_in_function(tmp_path: Path):
    src = "def f():\n    if True:\n        pass\n    else:\n        pass\n"
    f = tmp_path / "nested.py"
    f.write_text(src, encoding="utf-8-sig")

    nodes = PyLogicBlockBuilder.build(f)
    kinds = _dump_kinds(nodes)
    assert PublicLogicKind.FUNCTION in kinds
    assert PublicLogicKind.BRANCH in kinds


def test_try_except_finally_chain(tmp_path: Path):
    src = (
        "try:\n"
        "    pass\n"
        "except Exception:\n"
        "    pass\n"
        "else:\n"
        "    pass\n"
        "finally:\n"
        "    pass\n"
    )
    f = tmp_path / "trychain.py"
    f.write_text(src, encoding="utf-8-sig")

    result = PyLogicBlockBuilder.build(f)
    kinds = _dump_kinds(result)
    assert kinds.count(PublicLogicKind.TRY) == 1


def test_nested_loops_and_with(tmp_path: Path):
    src = (
        "for i in range(1):\n"
        "    with open('a') as f:\n"
        "        while True:\n"
        "            pass\n"
    )
    f = tmp_path / "loops.py"
    f.write_text(src, encoding="utf-8-sig")

    result = PyLogicBlockBuilder.build(f)
    kinds = _dump_kinds(result)

    assert PublicLogicKind.LOOP in kinds
    assert PublicLogicKind.WITH in kinds


def test_class_with_function(tmp_path: Path):
    src = "class X:\n    def f(self):\n        pass\n"
    f = tmp_path / "cls.py"
    f.write_text(src, encoding="utf-8-sig")

    result = PyLogicBlockBuilder.build(f)
    kinds = _dump_kinds(result)

    assert kinds.count(PublicLogicKind.CLASS) == 1
    assert kinds.count(PublicLogicKind.FUNCTION) == 1


def test_tree_is_pure_public_nodes(tmp_path: Path):
    src = "def f():\n    pass\n"
    f = tmp_path / "pure.py"
    f.write_text(src, encoding="utf-8-sig")

    result = PyLogicBlockBuilder.build(f)

    def check_pure(nodes):
        for n in nodes:
            assert isinstance(n, PublicLogicNode)
            assert all(isinstance(c, PublicLogicNode) for c in n.children)
            check_pure(n.children)

    check_pure(result)


def test_empty_file_returns_empty_list(tmp_path: Path):
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8-sig")
    result = PyLogicBlockBuilder.build(f)
    assert isinstance(result, list)
    assert result == []


def test_single_decorator_function(tmp_path: Path):
    src = "@dec\ndef f():\n    pass\n"
    f = tmp_path / "dec1.py"
    f.write_text(src, encoding="utf-8-sig")

    res = PyLogicBlockBuilder.build(f)
    kinds = _flatten_kinds(res)
    assert PublicLogicKind.FUNCTION in kinds
    assert _kinds_sequence(res) == [PublicLogicKind.FUNCTION]


def test_multiple_decorators_function_and_class(tmp_path: Path):
    src = "@d1\n@d2\ndef f():\n    pass\n\n@dc\nclass C:\n    pass\n"
    f = tmp_path / "dec_multi.py"
    f.write_text(src, encoding="utf-8-sig")

    res = PyLogicBlockBuilder.build(f)
    seq = _kinds_sequence(res)
    assert seq == [PublicLogicKind.FUNCTION, PublicLogicKind.CLASS]


def test_long_if_elif_chain(tmp_path: Path):
    src = "if a:\n    pass\nelif b:\n    pass\nelif c:\n    pass\nelse:\n    pass\n"
    f = tmp_path / "elif_chain.py"
    f.write_text(src, encoding="utf-8-sig")

    res = PyLogicBlockBuilder.build(f)
    assert len(res) == 1
    assert res[0].kind == PublicLogicKind.BRANCH


@pytest.mark.parametrize(
    "src",
    [
        # try/except
        ("try:\n    pass\nexcept Exception:\n    pass\n"),
        # try/except/else
        ("try:\n    pass\nexcept Exception:\n    pass\nelse:\n    pass\n"),
        # try/finally
        ("try:\n    pass\nfinally:\n    pass\n"),
        # try/except/else/finally
        (
            "try:\n"
            "    pass\n"
            "except Exception:\n"
            "    pass\n"
            "else:\n"
            "    pass\n"
            "finally:\n"
            "    pass\n"
        ),
    ],
)
def test_try_combinations_valid(tmp_path: Path, src: str):
    f = tmp_path / "try_ok.py"
    f.write_text(src, encoding="utf-8-sig")

    res = PyLogicBlockBuilder.build(f)
    kinds = _flatten_kinds(res)
    assert kinds.count(PublicLogicKind.TRY) == 1


def test_top_level_sibling_order(tmp_path: Path):
    src = """
def f():
    pass

class C:
    pass

if True:
    pass

for i in range(1):
    pass

try:
   pass

except Exception:
   pass

finally:
    pass

with open('a') as f:
    pass
"""
    f = tmp_path / "siblings.py"
    f.write_text(src, encoding="utf-8-sig")

    res = PyLogicBlockBuilder.build(f)
    seq = _kinds_sequence(res)
    assert seq == [
        PublicLogicKind.FUNCTION,
        PublicLogicKind.CLASS,
        PublicLogicKind.BRANCH,
        PublicLogicKind.LOOP,
        PublicLogicKind.TRY,
        PublicLogicKind.WITH,
    ]


def test_deep_nesting_path(tmp_path: Path):
    src = (
        "class Outer:\n"
        "    def f(self):\n"
        "        try:\n"
        "            while True:\n"
        "                if x:\n"
        "                    with open('a') as f:\n"
        "                        pass\n"
        "        except Exception:\n"
        "            pass\n"
    )
    f = tmp_path / "deep.py"
    f.write_text(src, encoding="utf-8-sig")

    res = PyLogicBlockBuilder.build(f)

    path = [
        PublicLogicKind.CLASS,
        PublicLogicKind.FUNCTION,
        PublicLogicKind.TRY,
        PublicLogicKind.LOOP,
        PublicLogicKind.BRANCH,
        PublicLogicKind.WITH,
    ]

    assert _has_path(res, path)


def test_every_public_has_origin(tmp_path):
    src = "def f():\n    if True:\n        pass\n"
    f = tmp_path / "orig.py"
    f.write_text(src, encoding="utf-8-sig")

    res = PyLogicBlockBuilder.build(f)

    def walk(nodes):
        for n in nodes:
            assert n.origin is not None, f"{n.kind} has no origin"
            walk(n.children)

    walk(res)


def test_visual_snapshot(tmp_path):
    src = """
def f():
    if x:
        for i in y:
            pass
"""
    f = tmp_path / "snap.py"
    f.write_text(src, encoding="utf-8-sig")

    result = PyLogicBlockBuilder.build(f)
    kinds = _dump_kinds(result)
    assert kinds == [
        PublicLogicKind.FUNCTION,
        PublicLogicKind.BRANCH,
        PublicLogicKind.LOOP,
        PublicLogicKind.STATEMENT,
    ]


@pytest.mark.parametrize(
    "src",
    [
        # else
        "else:\n    pass\n",
        # elif x
        "elif True:\n    pass\n",
        # exept
        "except Exception:\n    pass\n",
        # finally
        "finally:\n    pass\n",
    ],
)
def test_illegal_headers_raise_syntaxerror(tmp_path: Path, src: str):
    f = tmp_path / "broken.py"
    f.write_text(src, encoding="utf-8-sig")
    with pytest.raises(SyntaxError):
        PyLogicBlockBuilder.build(f)


def test_empty_block_raises(tmp_path: Path):
    f = tmp_path / "empty_block.py"
    f.write_text("def f():\n    ", encoding="utf-8-sig")
    with pytest.raises(SyntaxError):
        PyLogicBlockBuilder.build(f)


def test_multiple_try_chains_separated(tmp_path: Path):
    src = (
        "try:\n    pass\nexcept Exception:\n    pass\n"
        "try:\n    pass\nfinally:\n    pass\n"
    )
    f = tmp_path / "multi_try.py"
    f.write_text(src, encoding="utf-8-sig")
    res = PyLogicBlockBuilder.build(f)
    kinds = _kinds_sequence(res)
    assert kinds.count(PublicLogicKind.TRY) == 2


def test_function_with_nested_try_and_with(tmp_path: Path):
    src = (
        "def f():\n"
        "    try:\n"
        "        with open('a'):\n"
        "            pass\n"
        "    except Exception:\n"
        "        pass\n"
    )
    f = tmp_path / "nested_try_with.py"
    f.write_text(src, encoding="utf-8-sig")
    res = PyLogicBlockBuilder.build(f)
    kinds = _dump_kinds(res)
    assert PublicLogicKind.FUNCTION in kinds
    assert PublicLogicKind.TRY in kinds
    assert PublicLogicKind.WITH in kinds


def test_try_else_without_except_raises(tmp_path: Path):
    src = "try:\n    pass\nelse:\n    pass\n"
    f = tmp_path / "try_bad_else.py"
    f.write_text(src, encoding="utf-8-sig")
    with pytest.raises(SyntaxError):
        PyLogicBlockBuilder.build(f)


def test_decorator_without_target_raises(tmp_path: Path):
    src = "@dec\nx = 42\n"
    f = tmp_path / "decorator_fail.py"
    f.write_text(src, encoding="utf-8-sig")
    with pytest.raises(SyntaxError):
        PyLogicBlockBuilder.build(f)


def test_full_complex_script_parses(tmp_path: Path):
    src = (
        "@outer\n"
        "class A:\n"
        "    def f(self):\n"
        "        if x:\n"
        "            try:\n"
        "                while True:\n"
        "                    with open('a') as f:\n"
        "                        pass\n"
        "            except Exception:\n"
        "                pass\n"
    )
    f = tmp_path / "mega.py"
    f.write_text(src, encoding="utf-8-sig")
    res = PyLogicBlockBuilder.build(f)
    path = [
        PublicLogicKind.CLASS,
        PublicLogicKind.FUNCTION,
        PublicLogicKind.BRANCH,
        PublicLogicKind.TRY,
        PublicLogicKind.LOOP,
        PublicLogicKind.WITH,
    ]
    assert _has_path(res, path)


def test_no_internal_nodes_leak(tmp_path: Path):
    f = tmp_path / "leak.py"
    f.write_text("def f():\n    pass\n", encoding="utf-8-sig")
    res = PyLogicBlockBuilder.build(f)

    def walk(nodes):
        for n in nodes:
            assert isinstance(n, PublicLogicNode)
            walk(n.children)

    walk(res)

def test_import_statements_detected(tmp_path: Path):
    src = "import os\nfrom sys import path as sys_path\n"
    f = tmp_path / "imports.py"
    f.write_text(src, encoding="utf-8-sig")

    res = PyLogicBlockBuilder.build(f)
    kinds = [n.kind for n in res]
    assert all(k is PublicLogicKind.IMPORT for k in kinds)

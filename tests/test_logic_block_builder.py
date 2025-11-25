from collections import Counter

import pytest

from py2glua.lang.parse.py_logic_block_builder import (
    PyLogicBlockBuilder,
    PyLogicKind,
    PyLogicNode,
)
from py2glua.lang.parse.py_parser import PyParser, RawNonTerminal, RawNonTerminalKind


# region Helpers
def _walk_kinds(nodes, flat=True):
    kinds = []
    for n in nodes:
        kinds.append(n.kind)
        if flat or n.children:
            kinds.extend(_walk_kinds(n.children, flat=flat))

    return kinds


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


def _collect_all_raws(nodes, attr="tokens"):
    out = []
    for n in nodes:
        out.append(n)
        if hasattr(n, "children"):
            out.extend(_collect_all_raws(n.children, attr))

        if hasattr(n, attr):
            out.extend(_collect_all_raws(getattr(n, attr), attr))

    return out


def _collect_all_raws_from_parser(src):
    return _collect_all_raws(PyParser.parse(src), attr="tokens")


def _collect_all_raws_from_logic(nodes):
    return _collect_all_raws(nodes, attr="origins")


def _only_nt(seq):
    return [x for x in seq if isinstance(x, RawNonTerminal)]


# endregion


@pytest.mark.parametrize(
    "src,expected_kind",
    [
        ("def f():\n    pass\n", PyLogicKind.FUNCTION),
        ("class C:\n    pass\n", PyLogicKind.CLASS),
        ("if True:\n    pass\n", PyLogicKind.BRANCH),
        ("while True:\n    pass\n", PyLogicKind.LOOP),
        ("for i in range(1):\n    pass\n", PyLogicKind.LOOP),
        ("try:\n    pass\nexcept Exception:\n    pass\n", PyLogicKind.TRY),
        ("with open('a') as f:\n    pass\n", PyLogicKind.WITH),
    ],
)
def test_basic_block_detection(src: str, expected_kind: PyLogicKind):
    result = PyLogicBlockBuilder.build(src)
    kinds = _walk_kinds(result)
    assert expected_kind in kinds


def test_nested_if_in_function():
    nodes = PyLogicBlockBuilder.build(
        "def f():\n    if True:\n        pass\n    else:\n        pass\n"
    )
    kinds = _walk_kinds(nodes)
    assert PyLogicKind.FUNCTION in kinds
    assert PyLogicKind.BRANCH in kinds


def test_try_except_finally_chain():
    result = PyLogicBlockBuilder.build(
        "try:\n    pass\nexcept Exception:\n    pass\nelse:\n    pass\nfinally:\n    pass\n"
    )
    kinds = _walk_kinds(result)
    assert kinds.count(PyLogicKind.TRY) == 1


def test_nested_loops_and_with():
    result = PyLogicBlockBuilder.build(
        "for i in range(1):\n    with open('a') as f:\n        while True:\n            pass\n"
    )
    kinds = _walk_kinds(result)
    assert PyLogicKind.LOOP in kinds
    assert PyLogicKind.WITH in kinds


def test_class_with_function():
    result = PyLogicBlockBuilder.build("class X:\n    def f(self):\n        pass\n")
    kinds = _walk_kinds(result)
    assert kinds.count(PyLogicKind.CLASS) == 1
    assert kinds.count(PyLogicKind.FUNCTION) == 1


def test_tree_is_pure_public_nodes():
    result = PyLogicBlockBuilder.build("def f():\n    pass\n")

    def check(nodes):
        for n in nodes:
            assert isinstance(n, PyLogicNode)
            assert all(isinstance(c, PyLogicNode) for c in n.children)
            check(n.children)

    check(result)


def test_empty_file_returns_empty_list():
    result = PyLogicBlockBuilder.build("")
    assert isinstance(result, list)
    assert result == []


def test_single_decorator_function():
    res = PyLogicBlockBuilder.build("@dec\ndef f():\n    pass\n")
    kinds = _walk_kinds(res)
    assert PyLogicKind.FUNCTION in kinds
    assert _kinds_sequence(res) == [PyLogicKind.FUNCTION]


def test_multiple_decorators_function_and_class():
    res = PyLogicBlockBuilder.build(
        "@d1\n@d2\ndef f():\n    pass\n\n@dc\nclass C:\n    pass\n"
    )
    seq = _kinds_sequence(res)
    assert seq == [PyLogicKind.FUNCTION, PyLogicKind.CLASS]


def test_long_if_elif_chain():
    res = PyLogicBlockBuilder.build(
        "if a:\n    pass\nelif b:\n    pass\nelif c:\n    pass\nelse:\n    pass\n"
    )
    assert len(res) == 1
    assert res[0].kind == PyLogicKind.BRANCH


@pytest.mark.parametrize(
    "src",
    [
        "try:\n    pass\nexcept Exception:\n    pass\n",
        "try:\n    pass\nexcept Exception:\n    pass\nelse:\n    pass\n",
        "try:\n    pass\nfinally:\n    pass\n",
        "try:\n    pass\nexcept Exception:\n    pass\nelse:\n    pass\nfinally:\n    pass\n",
    ],
)
def test_try_combinations_valid(src: str):
    res = PyLogicBlockBuilder.build(src)
    kinds = _walk_kinds(res)
    assert kinds.count(PyLogicKind.TRY) == 1


def test_top_level_sibling_order():
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
    res = PyLogicBlockBuilder.build(src)
    seq = _kinds_sequence(res)
    assert seq == [
        PyLogicKind.FUNCTION,
        PyLogicKind.CLASS,
        PyLogicKind.BRANCH,
        PyLogicKind.LOOP,
        PyLogicKind.TRY,
        PyLogicKind.WITH,
    ]


def test_deep_nesting_path():
    res = PyLogicBlockBuilder.build(
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
    path = [
        PyLogicKind.CLASS,
        PyLogicKind.FUNCTION,
        PyLogicKind.TRY,
        PyLogicKind.LOOP,
        PyLogicKind.BRANCH,
        PyLogicKind.WITH,
    ]
    assert _has_path(res, path)


def test_every_public_has_origin():
    res = PyLogicBlockBuilder.build("def f():\n    if True:\n        pass\n")

    def walk(nodes):
        for n in nodes:
            assert n.origins, f"{n.kind} has no origins"
            walk(n.children)

    walk(res)


def test_visual_snapshot():
    result = PyLogicBlockBuilder.build(
        "def f():\n    if x:\n        for i in y:\n            pass\n"
    )
    kinds = _walk_kinds(result)
    assert kinds == [
        PyLogicKind.FUNCTION,
        PyLogicKind.BRANCH,
        PyLogicKind.LOOP,
        PyLogicKind.PASS,
    ]


@pytest.mark.parametrize(
    "src",
    [
        "else:\n    pass\n",
        "elif True:\n    pass\n",
        "except Exception:\n    pass\n",
        "finally:\n    pass\n",
    ],
)
def test_illegal_headers_raise_syntaxerror(src: str):
    with pytest.raises(SyntaxError):
        PyLogicBlockBuilder.build(src)


def test_empty_block_raises():
    with pytest.raises(SyntaxError):
        PyLogicBlockBuilder.build("def f():\n    ")


def test_multiple_try_chains_separated():
    res = PyLogicBlockBuilder.build(
        "try:\n    pass\nexcept Exception:\n    pass\n"
        "try:\n    pass\nfinally:\n    pass\n"
    )
    kinds = _kinds_sequence(res)
    assert kinds.count(PyLogicKind.TRY) == 2


def test_function_with_nested_try_and_with():
    res = PyLogicBlockBuilder.build(
        "def f():\n"
        "    try:\n"
        "        with open('a'):\n"
        "            pass\n"
        "    except Exception:\n"
        "        pass\n"
    )
    kinds = _walk_kinds(res)
    assert PyLogicKind.FUNCTION in kinds
    assert PyLogicKind.TRY in kinds
    assert PyLogicKind.WITH in kinds


def test_try_else_without_except_raises():
    with pytest.raises(SyntaxError):
        PyLogicBlockBuilder.build("try:\n    pass\nelse:\n    pass\n")


def test_decorator_without_target_raises():
    with pytest.raises(SyntaxError):
        PyLogicBlockBuilder.build("@dec\nx = 42\n")


def test_full_complex_script_parses():
    res = PyLogicBlockBuilder.build(
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
    path = [
        PyLogicKind.CLASS,
        PyLogicKind.FUNCTION,
        PyLogicKind.BRANCH,
        PyLogicKind.TRY,
        PyLogicKind.LOOP,
        PyLogicKind.WITH,
    ]
    assert _has_path(res, path)


def test_no_internal_nodes_leak():
    res = PyLogicBlockBuilder.build("def f():\n    pass\n")

    def walk(nodes):
        for n in nodes:
            assert isinstance(n, PyLogicNode)
            walk(n.children)

    walk(res)


def test_import_statements_detected():
    res = PyLogicBlockBuilder.build("import os\nfrom sys import path as sys_path\n")
    kinds = [n.kind for n in res]
    assert all(k is PyLogicKind.IMPORT for k in kinds)


def test_single_comment_is_detected():
    res = PyLogicBlockBuilder.build("# hello world\nx = 1\n")
    kinds = [n.kind for n in res]
    assert kinds == [PyLogicKind.COMMENT, PyLogicKind.STATEMENT]


def test_multiple_consecutive_comments_merge():
    res = PyLogicBlockBuilder.build("# a\n# b\n# c\nx = 1\n")
    kinds = [n.kind for n in res]
    assert kinds == [PyLogicKind.COMMENT, PyLogicKind.STATEMENT]
    comment_node = res[0]
    assert comment_node.origins
    assert all(r.kind.name == "COMMENT" for r in comment_node.origins)
    assert len(comment_node.origins) == 3


def test_comment_between_blocks_not_merged():
    res = PyLogicBlockBuilder.build(
        "def f():\n    pass\n# a\n# b\nclass C:\n    pass\n"
    )
    seq = _kinds_sequence(res)
    assert seq == [PyLogicKind.FUNCTION, PyLogicKind.COMMENT, PyLogicKind.CLASS]


def test_docstring_promoted_to_comment():
    res = PyLogicBlockBuilder.build('"""Docstring"""\nprint(1)\n')
    seq = _kinds_sequence(res)
    assert seq == [PyLogicKind.COMMENT, PyLogicKind.STATEMENT]


def test_docstring_and_comment_merge():
    res = PyLogicBlockBuilder.build('"""Docstring"""\n# comment\nprint(1)\n')
    seq = _kinds_sequence(res)
    assert seq == [PyLogicKind.COMMENT, PyLogicKind.STATEMENT]
    merged = res[0]
    assert merged.origins
    assert {r.kind.name for r in merged.origins} == {"DOCSTRING", "COMMENT"}
    assert len(merged.origins) == 2


def test_no_raw_nodes_lost_in_simple_file():
    src = (
        "@d1\n@d2\ndef f():\n"
        "    # inside\n"
        "    if True:\n        pass\n"
        "    else:\n        pass\n"
        "class C:\n    pass\n"
    )

    parsed_raws = _collect_all_raws_from_parser(src)
    logic_raws = _collect_all_raws_from_logic(PyLogicBlockBuilder.build(src))

    parsed_nt = _only_nt(parsed_raws)
    logic_nt = _only_nt(logic_raws)

    parsed_counts = Counter(r.kind for r in parsed_nt)
    logic_counts = Counter(r.kind for r in logic_nt)

    IGNORED_KINDS = {RawNonTerminalKind.BLOCK}
    missing = {
        k.name: v
        for k, v in (parsed_counts - logic_counts).items()
        if k not in IGNORED_KINDS
    }
    extras = {
        k.name: v
        for k, v in (logic_counts - parsed_counts).items()
        if k not in IGNORED_KINDS
    }

    assert not missing, f"Lost kinds: {missing}"
    assert not extras, f"Unexpected extra kinds in logic: {extras}"


def test_decorators_and_comments_survive():
    src = "# a\n# b\n@outer\n@inner\ndef f():\n    '''doc'''\n    pass\n"
    all_logic_raws = _collect_all_raws_from_logic(PyLogicBlockBuilder.build(src))
    kinds = [r.kind.name for r in all_logic_raws]
    assert "COMMENT" in kinds
    assert "DECORATORS" in kinds
    assert "DOCSTRING" in kinds


def test_branch_and_try_headers_preserved():
    src = (
        "if a:\n    pass\nelif b:\n    pass\nelse:\n    pass\n"
        "try:\n    pass\nexcept Exception:\n    pass\nelse:\n    pass\nfinally:\n    pass\n"
    )
    raws = _collect_all_raws_from_logic(PyLogicBlockBuilder.build(src))
    kinds = [r.kind.name for r in raws]

    for k in ("IF", "ELIF", "ELSE"):
        assert k in kinds, f"{k} header lost"

    for k in ("TRY", "EXCEPT", "ELSE", "FINALLY"):
        assert k in kinds, f"{k} header lost in try-chain"

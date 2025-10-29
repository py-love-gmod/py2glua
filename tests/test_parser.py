import textwrap

import pytest

from py2glua.lang.parser import Parser, RawNode, RawNodeKind


def _treeify(nodes):
    out = []
    for n in nodes:
        if n.kind == RawNodeKind.BLOCK:
            children = [t for t in n.tokens if isinstance(t, RawNode)]
            out.append(["BLOCK", _treeify(children)])

        else:
            out.append(n.kind.name)

    return out


def test_non_constructs_are_ignored_at_toplevel():
    src = "x = 1\ny = x + 2\nprint(y)\n"
    nodes = Parser.parse(src)
    assert nodes == [], (
        "Non-header statements should be ignored at top-level in raw phase"
    )


def test_simple_function_header_and_block():
    src = textwrap.dedent(
        """
        def f(a, b=1):
            pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "FUNCTION",
        ["BLOCK", []],
    ]


def test_nested_if_inside_function():
    src = textwrap.dedent(
        """
        def f():
            if (a and b) or (c and (d or e)):
                pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "FUNCTION",
        [
            "BLOCK",
            [
                "IF",
                ["BLOCK", []],
            ],
        ],
    ]


def test_class_header_and_block_with_method():
    src = textwrap.dedent(
        """
        class X(Base):
            def m(self):
                if ok:
                    pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "CLASS",
        [
            "BLOCK",
            [
                "FUNCTION",
                ["BLOCK", ["IF", ["BLOCK", []]]],
            ],
        ],
    ]


def test_decorators_before_function():
    src = textwrap.dedent(
        """
        @decor
        @decor2(x, y)
        def f():
            pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "DECORATORS",
        "DECORATORS",
        "FUNCTION",
        ["BLOCK", []],
    ]


def test_import_single_and_multiline_paren():
    src = textwrap.dedent(
        r"""
        import os, sys
        from pkg.subpkg import (
            a,
            b,
        )
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "IMPORT",
        "IMPORT",
    ]


def test_import_with_backslash_continuation():
    src = textwrap.dedent(
        r"""
        import os, \
               sys, \
               json
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == ["IMPORT"]


def test_del_multiline_with_brackets():
    src = textwrap.dedent(
        """
        del arr[
            i:j
        ]
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == ["DEL"]


def test_try_except_finally_structure():
    src = textwrap.dedent(
        """
        try:
            pass
        except ValueError:
            pass
        finally:
            pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "TRY",
        ["BLOCK", []],
        "EXCEPT",
        ["BLOCK", []],
        "FINALLY",
        ["BLOCK", []],
    ]


def test_if_elif_else_structure():
    src = textwrap.dedent(
        """
        if a:
            pass
        elif b:
            pass
        else:
            pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "IF",
        ["BLOCK", []],
        "ELIF",
        ["BLOCK", []],
        "ELSE",
        ["BLOCK", []],
    ]


def test_while_with_else():
    src = textwrap.dedent(
        """
        while cond:
            pass
        else:
            pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "WHILE",
        ["BLOCK", []],
        "ELSE",
        ["BLOCK", []],
    ]


def test_for_with_else_nested():
    src = textwrap.dedent(
        """
        for i in range(3):
            if i % 2:
                pass
        else:
            pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "FOR",
        [
            "BLOCK",
            [
                "IF",
                ["BLOCK", []],
            ],
        ],
        "ELSE",
        ["BLOCK", []],
    ]


def test_with_multiline_header():
    src = textwrap.dedent(
        """
        with (
            ctx1 as a,
            ctx2 as b
        ):
            pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "WITH",
        ["BLOCK", []],
    ]


def test_nested_blocks_three_levels():
    src = textwrap.dedent(
        """
        def f():
            if x:
                for i in range(2):
                    while cond:
                        pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "FUNCTION",
        [
            "BLOCK",
            [
                "IF",
                [
                    "BLOCK",
                    [
                        "FOR",
                        [
                            "BLOCK",
                            [
                                "WHILE",
                                ["BLOCK", []],
                            ],
                        ],
                    ],
                ],
            ],
        ],
    ]


@pytest.mark.parametrize("kw", ["global", "nonlocal", "async", "await", "yield"])
def test_forbidden_keywords_raise(kw):
    src = f"{kw} x\n"
    with pytest.raises(SyntaxError):
        Parser.parse(src)


def test_async_def_is_forbidden_due_to_async_token():
    src = textwrap.dedent(
        """
        async def f():
            pass
        """
    )
    with pytest.raises(SyntaxError):
        Parser.parse(src)


def test_invalid_syntax_rejected_by_ast_parse():
    src = "if x\n    pass\n"
    with pytest.raises(SyntaxError):
        Parser.parse(src)

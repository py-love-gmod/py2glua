import textwrap
import tokenize

import pytest

from py2glua.lang.parser import Parser, RawNode, RawNodeKind


# region Helpers
def _treeify(nodes):
    out = []
    for n in nodes:
        if n.kind == RawNodeKind.BLOCK:
            children = [t for t in n.tokens if isinstance(t, RawNode)]
            out.append(["BLOCK", _treeify(children)])

        else:
            out.append(n.kind.name)

    return out


def _only_kinds(nodes):
    out = []
    for n in nodes:
        if n.kind == RawNodeKind.BLOCK:
            children = [t for t in n.tokens if isinstance(t, RawNode)]
            out.append(["BLOCK", _only_kinds(children)])

        else:
            out.append(n.kind.name)

    return out


def _other_tokens(node: RawNode):
    assert node.kind == RawNodeKind.OTHER
    toks = []
    for t in node.tokens:
        if isinstance(t, RawNode):
            continue

        if t.type in (tokenize.NL, tokenize.NEWLINE):
            continue

        toks.append(t.string)

    return toks


# endregion


# region Basic structure
def test_empty_file_has_no_nodes():
    assert Parser.parse("") == []


def test_whitespace_only_has_no_nodes():
    assert Parser.parse("\n \n\t\n") == []


def test_non_constructs_are_ignored_at_toplevel():
    src = "x = 1\ny = x + 2\nprint(y)\n"
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes] == [
        RawNodeKind.OTHER,
        RawNodeKind.OTHER,
        RawNodeKind.OTHER,
    ]


def test_toplevel_semicolons_single_other():
    nodes = Parser.parse("a=1; b=2\n")
    assert len(nodes) == 1 and nodes[0].kind == RawNodeKind.OTHER
    assert _other_tokens(nodes[0]) == ["a", "=", "1", ";", "b", "=", "2"]


# endregion


# region Imports
@pytest.mark.parametrize(
    "src, expected",
    [
        (
            textwrap.dedent(
                r"""
                import os, sys
                from pkg.subpkg import (
                    a,
                    b,
                )
                """
            ),
            ["IMPORT", "IMPORT"],
        ),
        (
            textwrap.dedent(
                r"""
                import os, \
                       sys, \
                       json
                """
            ),
            ["IMPORT"],
        ),
        (
            "from __future__ import annotations as ann\n",
            ["IMPORT"],
        ),
    ],
)
def test_import_variants(src, expected):
    assert _treeify(Parser.parse(src)) == expected


# endregion


# region Headers & blocks
def test_simple_function_header_and_block():
    src = textwrap.dedent(
        """
        def f(a, b=1):
            pass
        """
    )
    assert _treeify(Parser.parse(src)) == ["FUNCTION", ["BLOCK", ["PASS"]]]


def test_class_header_and_block_with_method():
    src = textwrap.dedent(
        """
        class X(Base):
            def m(self):
                if ok:
                    pass
        """
    )
    assert _treeify(Parser.parse(src)) == [
        "CLASS",
        ["BLOCK", ["FUNCTION", ["BLOCK", ["IF", ["BLOCK", ["PASS"]]]]]],
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
    assert _treeify(Parser.parse(src)) == [
        "FUNCTION",
        [
            "BLOCK",
            ["IF", ["BLOCK", ["FOR", ["BLOCK", ["WHILE", ["BLOCK", ["PASS"]]]]]]],
        ],
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
    assert _treeify(Parser.parse(src)) == [
        "IF",
        ["BLOCK", ["PASS"]],
        "ELIF",
        ["BLOCK", ["PASS"]],
        "ELSE",
        ["BLOCK", ["PASS"]],
    ]


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
    assert _treeify(Parser.parse(src)) == [
        "TRY",
        ["BLOCK", ["PASS"]],
        "EXCEPT",
        ["BLOCK", ["PASS"]],
        "FINALLY",
        ["BLOCK", ["PASS"]],
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
    assert _treeify(Parser.parse(src)) == [
        "WHILE",
        ["BLOCK", ["PASS"]],
        "ELSE",
        ["BLOCK", ["PASS"]],
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
    assert _treeify(Parser.parse(src)) == [
        "FOR",
        ["BLOCK", ["IF", ["BLOCK", ["PASS"]]]],
        "ELSE",
        ["BLOCK", ["PASS"]],
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
    assert _treeify(Parser.parse(src)) == ["WITH", ["BLOCK", ["PASS"]]]


# endregion


# region Inline suites
@pytest.mark.parametrize(
    "src, expected",
    [
        ("if x: pass\n", ["IF", ["BLOCK", ["OTHER"]]]),
        ("while ok: step()\n", ["WHILE", ["BLOCK", ["OTHER"]]]),
        ("for i in (1,2): use(i)\n", ["FOR", ["BLOCK", ["OTHER"]]]),
        ("def f(): return 1\n", ["FUNCTION", ["BLOCK", ["OTHER"]]]),
        ("class C: pass\n", ["CLASS", ["BLOCK", ["OTHER"]]]),
    ],
)
def test_inline_suites(src, expected):
    assert _treeify(Parser.parse(src)) == expected


def test_header_with_parens_and_backslash_before_colon_inline_body():
    src = textwrap.dedent(
        """
        if (a and (b or c)) \\
           and d: pass
        """
    )
    assert _treeify(Parser.parse(src)) == ["IF", ["BLOCK", ["OTHER"]]]


def test_inline_else_comes_on_next_line_is_ok():
    src = textwrap.dedent(
        """
        if a: x = 1
        else:
            y = 2
        """
    )
    assert _treeify(Parser.parse(src)) == [
        "IF",
        ["BLOCK", ["OTHER"]],
        "ELSE",
        ["BLOCK", ["OTHER"]],
    ]


def test_semicolons_same_line_single_other():
    src = textwrap.dedent(
        """
        def f():
            a=1; b=2; c=3
        """
    )
    nodes = Parser.parse(src)
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert len(inner) == 1 and inner[0].kind == RawNodeKind.OTHER
    assert _other_tokens(inner[0]) == [
        "a",
        "=",
        "1",
        ";",
        "b",
        "=",
        "2",
        ";",
        "c",
        "=",
        "3",
    ]


# region OTHER tokenization details
def test_other_at_toplevel_assignment_and_call():
    src = "x = 1\nprint(x)\n"
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes] == [RawNodeKind.OTHER, RawNodeKind.OTHER]
    assert _other_tokens(nodes[0]) == ["x", "=", "1"]
    assert _other_tokens(nodes[1]) == ["print", "(", "x", ")"]


@pytest.mark.parametrize(
    "src, expected_tokens",
    [
        (
            textwrap.dedent(
                r"""
                def f():
                    x = 1 + \
                        2 + \
                        3
                """
            ),
            ["x", "=", "1", "+", "2", "+", "3"],
        ),
        (
            textwrap.dedent(
                """
                def f():
                    y = (
                        10 +
                        20 +
                        30
                    )
                """
            ),
            ["y", "=", "(", "10", "+", "20", "+", "30", ")"],
        ),
    ],
)
def test_other_multiline_forms(src, expected_tokens):
    nodes = Parser.parse(src)
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert len(inner) == 1 and inner[0].kind == RawNodeKind.OTHER
    assert _other_tokens(inner[0]) == expected_tokens


def test_parenthesized_tuple_assignment_multiline_in_block():
    src = textwrap.dedent(
        """
        def f():
            x = (
                1,
                2,
            )
        """
    )
    nodes = Parser.parse(src)
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert len(inner) == 1 and inner[0].kind == RawNodeKind.OTHER


# endregion


# region Decorators
@pytest.mark.parametrize(
    "src, expected",
    [
        (
            textwrap.dedent(
                """
                @decor
                @decor2(x, y)
                def f():
                    pass
                """
            ),
            ["DECORATORS", "DECORATORS", "FUNCTION", ["BLOCK", ["PASS"]]],
        ),
        (
            textwrap.dedent(
                """
                @decor(
                    1,
                    2
                )
                def f():
                    pass
                """
            ),
            ["DECORATORS", "FUNCTION", ["BLOCK", ["PASS"]]],
        ),
    ],
)
def test_decorators_then_function(src, expected):
    assert _only_kinds(Parser.parse(src)) == expected


# endregion


# region Comments
def test_comment_line_is_other_and_blank_lines_skipped():
    src = textwrap.dedent(
        """
        # just a comment

        x = 1  # inline comment
        """
    )
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes] == [
        RawNodeKind.COMMENT,
        RawNodeKind.OTHER,
        RawNodeKind.COMMENT,
    ]
    cmt = nodes[0]
    assert any(getattr(t, "type", None) == tokenize.COMMENT for t in cmt.tokens)
    assert _other_tokens(nodes[1])[:3] == ["x", "=", "1"]


def test_comment_only_file_becomes_one_comment():
    nodes = Parser.parse("# hi\n")
    assert len(nodes) == 1 and nodes[0].kind == RawNodeKind.COMMENT
    assert any(getattr(t, "type", None) == tokenize.COMMENT for t in nodes[0].tokens)


def test_comment_nodes_present():
    src = """# shebang or comment
x = 1
# inner comment
y = 2
    """
    nodes = Parser.parse(src)
    assert nodes[0].kind == RawNodeKind.COMMENT


# endregion


# region Docstrings
def test_module_docstring():
    nodes = Parser.parse('"""hello\\nworld"""\n')
    assert len(nodes) == 1 and nodes[0].kind == RawNodeKind.DOCSTRING
    assert any(getattr(t, "type", None) == tokenize.STRING for t in nodes[0].tokens)


def test_module_docstring_promoted_then_other():
    nodes = Parser.parse('"""hello"""\nx = 1\n')
    assert nodes[0].kind == RawNodeKind.DOCSTRING
    assert any(n.kind == RawNodeKind.OTHER for n in nodes[1:])


def test_docstring_then_def_sequence():
    src = '"""mod doc"""\n\ndef f():\n    pass\n'
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes[:3]] == [
        RawNodeKind.DOCSTRING,
        RawNodeKind.FUNCTION,
        RawNodeKind.BLOCK,
    ]


def test_function_docstring_promoted():
    src = 'def f():\n    """doc"""\n    x = 1\n'
    nodes = Parser.parse(src)
    assert nodes[0].kind == RawNodeKind.FUNCTION
    assert nodes[1].kind == RawNodeKind.BLOCK
    inner = [n for n in nodes[1].tokens if isinstance(n, type(nodes[0]))]
    assert inner and inner[0].kind == RawNodeKind.DOCSTRING


# endregion


# region Mixed / edge cases
def test_other_inside_function_body():
    src = textwrap.dedent(
        """
        def f():
            x = 1
            print("hi")
        """
    )
    nodes = Parser.parse(src)
    assert nodes[0].kind == RawNodeKind.FUNCTION
    assert nodes[1].kind == RawNodeKind.BLOCK
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert [n.kind for n in inner] == [RawNodeKind.OTHER, RawNodeKind.OTHER]
    assert _other_tokens(inner[0]) == ["x", "=", "1"]
    assert _other_tokens(inner[1]) == ["print", "(", '"hi"', ")"]


def test_nested_inline_if_in_block():
    src = textwrap.dedent(
        """
        def f():
            if a: return 1
        """
    )
    assert _treeify(Parser.parse(src)) == [
        "FUNCTION",
        ["BLOCK", ["IF", ["BLOCK", ["OTHER"]]]],
    ]


def test_nested_dedent_after_block_no_spurious_other():
    src = textwrap.dedent(
        """
        def f():
            if x:
                pass
        y = 1
        """
    )
    nodes = Parser.parse(src)
    assert nodes[0].kind == RawNodeKind.FUNCTION
    assert nodes[1].kind == RawNodeKind.BLOCK
    assert nodes[2].kind == RawNodeKind.OTHER


def test_block_with_blank_lines_inside():
    src = textwrap.dedent(
        """
        def f():

            x = 1

        """
    )
    nodes = Parser.parse(src)
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert [n.kind for n in inner] == [RawNodeKind.OTHER]
    assert _other_tokens(inner[0]) == ["x", "=", "1"]


def test_del_multiline_with_brackets():
    src = textwrap.dedent(
        """
        del arr[
            i:j
        ]
        """
    )
    assert _treeify(Parser.parse(src)) == ["DEL"]


def test_try_except_finally_with_other():
    src = textwrap.dedent(
        """
        try:
            risky()
        except ValueError:
            log("bad")
        finally:
            cleanup()
        """
    )
    assert _only_kinds(Parser.parse(src)) == [
        "TRY",
        ["BLOCK", ["OTHER"]],
        "EXCEPT",
        ["BLOCK", ["OTHER"]],
        "FINALLY",
        ["BLOCK", ["OTHER"]],
    ]


def test_while_else_with_other():
    src = textwrap.dedent(
        """
        while cond:
            step()
        else:
            done()
        """
    )
    assert _only_kinds(Parser.parse(src)) == [
        "WHILE",
        ["BLOCK", ["OTHER"]],
        "ELSE",
        ["BLOCK", ["OTHER"]],
    ]


# endregion


# region Errors & forbidden
@pytest.mark.parametrize("kw", ["global", "nonlocal", "async", "await", "yield"])
def test_forbidden_keywords_raise(kw):
    with pytest.raises(SyntaxError):
        Parser.parse(f"{kw} x\n")


def test_async_def_is_forbidden_due_to_async_token():
    src = textwrap.dedent(
        """
        async def f():
            pass
        """
    )
    with pytest.raises(SyntaxError):
        Parser.parse(src)


def test_invalid_syntax_rejected_by_compile_exec():
    with pytest.raises(SyntaxError):
        Parser.parse("if x\n    pass\n")


# endregion

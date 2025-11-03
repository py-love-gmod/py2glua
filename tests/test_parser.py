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


# endregion


def test_non_constructs_are_ignored_at_toplevel():
    src = "x = 1\ny = x + 2\nprint(y)\n"
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes] == [
        RawNodeKind.OTHER,
        RawNodeKind.OTHER,
        RawNodeKind.OTHER,
    ]


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
        ["BLOCK", ["OTHER"]],
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
                ["BLOCK", ["OTHER"]],
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
                [
                    "BLOCK",
                    [
                        "IF",
                        ["BLOCK", ["OTHER"]],
                    ],
                ],
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
        ["BLOCK", ["OTHER"]],
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
        ["BLOCK", ["OTHER"]],
        "EXCEPT",
        ["BLOCK", ["OTHER"]],
        "FINALLY",
        ["BLOCK", ["OTHER"]],
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
        ["BLOCK", ["OTHER"]],
        "ELIF",
        ["BLOCK", ["OTHER"]],
        "ELSE",
        ["BLOCK", ["OTHER"]],
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
        ["BLOCK", ["OTHER"]],
        "ELSE",
        ["BLOCK", ["OTHER"]],
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
                ["BLOCK", ["OTHER"]],
            ],
        ],
        "ELSE",
        ["BLOCK", ["OTHER"]],
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
        ["BLOCK", ["OTHER"]],
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
                                ["BLOCK", ["OTHER"]],
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


def test_other_at_toplevel_assignment_and_call():
    src = "x = 1\nprint(x)\n"
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes] == [RawNodeKind.OTHER, RawNodeKind.OTHER]
    assert _other_tokens(nodes[0]) == ["x", "=", "1"]
    assert _other_tokens(nodes[1]) == ["print", "(", "x", ")"]


def test_module_docstring_is_other():
    src = '"""hello\\nworld"""\n'
    nodes = Parser.parse(src)
    assert len(nodes) == 1 and nodes[0].kind == RawNodeKind.OTHER
    assert any(getattr(t, "type", None) == tokenize.STRING for t in nodes[0].tokens)


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


def test_pass_and_return_become_other():
    src = textwrap.dedent(
        """
        def f():
            pass
            return 42
        """
    )
    nodes = Parser.parse(src)
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert [n.kind for n in inner] == [RawNodeKind.OTHER, RawNodeKind.OTHER]
    assert _other_tokens(inner[0]) == ["pass"]
    assert _other_tokens(inner[1]) == ["return", "42"]


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
    toks = _other_tokens(inner[0])
    assert toks == ["a", "=", "1", ";", "b", "=", "2", ";", "c", "=", "3"]


def test_other_with_backslash_continuation():
    src = textwrap.dedent(
        r"""
        def f():
            x = 1 + \
                2 + \
                3
        """
    )
    nodes = Parser.parse(src)
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert len(inner) == 1 and inner[0].kind == RawNodeKind.OTHER
    toks = _other_tokens(inner[0])
    assert toks == ["x", "=", "1", "+", "2", "+", "3"]


def test_other_with_parentheses_multiline():
    src = textwrap.dedent(
        """
        def f():
            y = (
                10 +
                20 +
                30
            )
        """
    )
    nodes = Parser.parse(src)
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert len(inner) == 1
    toks = _other_tokens(inner[0])
    assert toks == ["y", "=", "(", "10", "+", "20", "+", "30", ")"]


def test_comment_line_is_other_and_blank_lines_skipped():
    src = textwrap.dedent(
        """
        # just a comment

        x = 1  # inline comment
        """
    )
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes] == [RawNodeKind.OTHER, RawNodeKind.OTHER]
    cmt = nodes[0]
    assert any(getattr(t, "type", None) == tokenize.COMMENT for t in cmt.tokens)
    toks = _other_tokens(nodes[1])
    assert toks[:3] == ["x", "=", "1"]


def test_if_else_with_other_bodies():
    src = textwrap.dedent(
        """
        if a:
            s = "A"
        else:
            s = "B"
        """
    )
    nodes = Parser.parse(src)
    assert _only_kinds(nodes) == [
        "IF",
        ["BLOCK", ["OTHER"]],
        "ELSE",
        ["BLOCK", ["OTHER"]],
    ]


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
    nodes = Parser.parse(src)
    assert _only_kinds(nodes) == [
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
    nodes = Parser.parse(src)
    assert _only_kinds(nodes) == [
        "WHILE",
        ["BLOCK", ["OTHER"]],
        "ELSE",
        ["BLOCK", ["OTHER"]],
    ]


def test_nested_blocks_with_other_and_headers():
    src = textwrap.dedent(
        """
        def f():
            if x:
                z = 1
                for i in range(2):
                    use(i)
            y = 3
        """
    )
    nodes = Parser.parse(src)
    assert nodes[0].kind == RawNodeKind.FUNCTION
    block = nodes[1]
    assert block.kind == RawNodeKind.BLOCK
    inner = [t for t in block.tokens if isinstance(t, RawNode)]
    kinds = [n.kind for n in inner]
    assert kinds == [RawNodeKind.IF, RawNodeKind.BLOCK, RawNodeKind.OTHER]
    if_body = [t for t in inner[1].tokens if isinstance(t, RawNode)]
    assert [n.kind for n in if_body] == [
        RawNodeKind.OTHER,
        RawNodeKind.FOR,
        RawNodeKind.BLOCK,
    ]
    for_body = [t for t in if_body[2].tokens if isinstance(t, RawNode)]
    assert [n.kind for n in for_body] == [RawNodeKind.OTHER]


def test_multiline_decorator_then_function():
    src = textwrap.dedent(
        """
        @decor(
            1,
            2
        )
        def f():
            pass
        """
    )
    nodes = Parser.parse(src)
    assert _only_kinds(nodes) == [
        "DECORATORS",
        "FUNCTION",
        ["BLOCK", ["OTHER"]],
    ]


def test_empty_file_has_no_nodes():
    src = ""
    nodes = Parser.parse(src)
    assert nodes == []


def test_whitespace_only_has_no_nodes():
    src = "\n \n\t\n"
    nodes = Parser.parse(src)
    assert nodes == []


def test_comment_only_file_becomes_one_other():
    src = "# hi\n"
    nodes = Parser.parse(src)
    assert len(nodes) == 1 and nodes[0].kind == RawNodeKind.OTHER
    assert any(getattr(t, "type", None) == tokenize.COMMENT for t in nodes[0].tokens)


def test_header_after_blank_lines_no_fake_other():
    src = "\n\n\ndef f():\n    pass\n"
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes] == [RawNodeKind.FUNCTION, RawNodeKind.BLOCK]
    assert [c.kind for c in [t for t in nodes[1].tokens if isinstance(t, RawNode)]] == [
        RawNodeKind.OTHER
    ]


def test_docstring_then_def_sequence():
    src = '"""mod doc"""\n\ndef f():\n    pass\n'
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes[:3]] == [
        RawNodeKind.OTHER,
        RawNodeKind.FUNCTION,
        RawNodeKind.BLOCK,
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


def test_toplevel_semicolons_single_other():
    src = "a=1; b=2\n"
    nodes = Parser.parse(src)
    assert len(nodes) == 1 and nodes[0].kind == RawNodeKind.OTHER
    toks = _other_tokens(nodes[0])
    assert toks == ["a", "=", "1", ";", "b", "=", "2"]


def test_from_future_import_and_aliases():
    src = "from __future__ import annotations as ann\n"
    nodes = Parser.parse(src)
    assert _treeify(nodes) == ["IMPORT"]


def test_with_multi_context_inline():
    src = "with a as x, b as y: do()\n"
    nodes = Parser.parse(src)
    assert [n.kind for n in nodes] == [RawNodeKind.WITH, RawNodeKind.BLOCK]
    inner = [t for t in nodes[1].tokens if isinstance(t, RawNode)]
    assert len(inner) == 1 and inner[0].kind == RawNodeKind.OTHER


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


def test_inline_suite_if_single_line():
    src = "if x: pass\n"
    nodes = Parser.parse(src)
    assert _treeify(nodes) == ["IF", ["BLOCK", ["OTHER"]]]


def test_inline_suite_while_and_for_single_line():
    src = "while ok: step()\nfor i in (1,2): use(i)\n"
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "WHILE",
        ["BLOCK", ["OTHER"]],
        "FOR",
        ["BLOCK", ["OTHER"]],
    ]


def test_inline_def_and_class_single_line():
    src = "def f(): return 1\nclass C: pass\n"
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "FUNCTION",
        ["BLOCK", ["OTHER"]],
        "CLASS",
        ["BLOCK", ["OTHER"]],
    ]


def test_inline_try_chain_single_line():
    src = "try: risky()\nexcept ValueError: handle()\nfinally: cleanup()\n"
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "TRY",
        ["BLOCK", ["OTHER"]],
        "EXCEPT",
        ["BLOCK", ["OTHER"]],
        "FINALLY",
        ["BLOCK", ["OTHER"]],
    ]


def test_inline_if_with_multiple_simple_stmts():
    src = "if x: a=1; b=2; c=3\n"
    nodes = Parser.parse(src)
    assert _treeify(nodes) == ["IF", ["BLOCK", ["OTHER"]]]
    block = nodes[1]
    inner_other = [t for t in block.tokens if isinstance(t, RawNode)][0]
    toks = _other_tokens(inner_other)
    assert toks == ["a", "=", "1", ";", "b", "=", "2", ";", "c", "=", "3"]


def test_nested_inline_if_in_block():
    src = textwrap.dedent(
        """
        def f():
            if a: return 1
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "FUNCTION",
        ["BLOCK", ["IF", ["BLOCK", ["OTHER"]]]],
    ]


def test_header_with_parens_and_backslash_before_colon_inline_body():
    src = textwrap.dedent(
        """
        if (a and (b or c)) \\
           and d: pass
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == ["IF", ["BLOCK", ["OTHER"]]]


def test_inline_with_multi_contexts_then_other():
    src = "with a as x, b as y: do();\n"
    nodes = Parser.parse(src)
    assert _treeify(nodes) == ["WITH", ["BLOCK", ["OTHER"]]]
    inner_other = [t for t in nodes[1].tokens if isinstance(t, RawNode)][0]
    toks = _other_tokens(inner_other)
    assert toks[:3] == ["do", "(", ")"]


def test_inline_else_comes_on_next_line_is_ok():
    src = textwrap.dedent(
        """
        if a: x = 1
        else:
            y = 2
        """
    )
    nodes = Parser.parse(src)
    assert _treeify(nodes) == [
        "IF",
        ["BLOCK", ["OTHER"]],
        "ELSE",
        ["BLOCK", ["OTHER"]],
    ]

import tokenize
from typing import Sequence

from ...parse import PyLogicKind, PyLogicNode
from ..build_context import build_block, build_expr
from ..ir_dataclass import PyIRClassDef, PyIRNode


def parse_bases(tokens: list[tokenize.TokenInfo]) -> list[list[tokenize.TokenInfo]]:
    bases: list[list[tokenize.TokenInfo]] = []

    depth = 0
    current: list[tokenize.TokenInfo] = []

    for t in tokens:
        if t.type == tokenize.OP and t.string == "(":
            depth = 1
            continue

        if depth == 0:
            continue

        if t.type == tokenize.OP and t.string == "(":
            depth += 1
            current.append(t)
            continue

        if t.type == tokenize.OP and t.string == ")":
            depth -= 1
            if depth == 0:
                if current:
                    bases.append(current)
                break
            current.append(t)
            continue

        if t.type == tokenize.OP and t.string == "," and depth == 1:
            bases.append(current)
            current = []
            continue

        current.append(t)

    if depth != 0:
        raise SyntaxError("Unclosed base class list")

    return bases


class ClassBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if node.kind is not PyLogicKind.CLASS:
            raise ValueError("ClassBuilder expects PyLogicKind.CLASS")

        if not node.origins:
            raise SyntaxError("Class node has no header")

        header = node.origins[0]

        tokens = [
            t
            for t in header.tokens
            if isinstance(t, tokenize.TokenInfo)
            and t.type
            not in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            )
        ]

        if len(tokens) < 2 or tokens[0].string != "class":
            raise SyntaxError("Invalid class declaration")

        name_tok = tokens[1]
        if name_tok.type != tokenize.NAME:
            raise SyntaxError("Expected class name")

        class_name = name_tok.string
        bases: list[PyIRNode] = []

        if any(t.type == tokenize.OP and t.string == "(" for t in tokens):
            for group in parse_bases(tokens):
                bases.append(build_expr(group))

        body = build_block(list(node.children))
        line, col = name_tok.start

        return [
            PyIRClassDef(
                line=line,
                offset=col,
                name=class_name,
                bases=bases,
                decorators=[],
                body=body,
            )
        ]

import tokenize
from typing import Sequence

from ...etc import TokenStream
from ...parse import PyLogicKind, PyLogicNode
from ..build_context import build_block
from ..ir_dataclass import PyIRFor, PyIRNode, PyIRWhile
from .statement_builder import StatementBuilder


class LoopBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if node.kind is not PyLogicKind.LOOP:
            raise ValueError("LoopBuilder expects PyLogicKind.LOOP")

        if not node.origins:
            raise ValueError("PyLogicNode.LOOP has no origins")

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

        if not tokens:
            raise SyntaxError("Empty loop header")

        first = tokens[0]
        if first.type != tokenize.NAME:
            raise SyntaxError("Invalid loop header")

        body = build_block(node.children)

        if first.string == "while":
            return [LoopBuilder._build_while(tokens, body)]

        if first.string == "for":
            return [LoopBuilder._build_for(tokens, body)]

        raise SyntaxError(f"Unknown loop type: {first.string!r}")

    # while <expr>:
    @staticmethod
    def _build_while(
        tokens: list[tokenize.TokenInfo],
        body: list[PyIRNode],
    ) -> PyIRWhile:
        if tokens[-1].type != tokenize.OP or tokens[-1].string != ":":
            raise SyntaxError("While header must end with ':'")

        expr_tokens = tokens[1:-1]
        if not expr_tokens:
            raise SyntaxError("Missing condition in while loop")

        stream = TokenStream(expr_tokens)
        test = StatementBuilder._parse_expression(stream)
        if not stream.eof():
            raise SyntaxError("Invalid condition in while loop")

        line, col = tokens[0].start

        return PyIRWhile(
            line=line,
            offset=col,
            test=test,
            body=body,
        )

    # for <target> in <iter>:
    @staticmethod
    def _build_for(
        tokens: list[tokenize.TokenInfo],
        body: list[PyIRNode],
    ) -> PyIRFor:
        if tokens[-1].type != tokenize.OP or tokens[-1].string != ":":
            raise SyntaxError("For header must end with ':'")

        # ищем `in` на верхнем уровне
        depth_paren = depth_brack = depth_brace = 0
        in_index: int | None = None

        for i, t in enumerate(tokens):
            if t.type == tokenize.OP:
                if t.string == "(":
                    depth_paren += 1

                elif t.string == ")":
                    depth_paren -= 1

                elif t.string == "[":
                    depth_brack += 1

                elif t.string == "]":
                    depth_brack -= 1

                elif t.string == "{":
                    depth_brace += 1

                elif t.string == "}":
                    depth_brace -= 1

            if (
                depth_paren == 0
                and depth_brack == 0
                and depth_brace == 0
                and t.type == tokenize.NAME
                and t.string == "in"
            ):
                in_index = i
                break

        if in_index is None:
            raise SyntaxError("Expected 'in' in for loop")

        target_tokens = tokens[1:in_index]
        iter_tokens = tokens[in_index + 1 : -1]

        if not target_tokens:
            raise SyntaxError("Missing target in for loop")

        if not iter_tokens:
            raise SyntaxError("Missing iterable in for loop")

        target_stream = TokenStream(target_tokens)
        target = StatementBuilder._parse_expression(target_stream)
        if not target_stream.eof():
            raise SyntaxError("Invalid target in for loop")

        iter_stream = TokenStream(iter_tokens)
        iter_expr = StatementBuilder._parse_expression(iter_stream)
        if not iter_stream.eof():
            raise SyntaxError("Invalid iterable in for loop")

        line, col = tokens[0].start

        return PyIRFor(
            line=line,
            offset=col,
            target=target,
            iter=iter_expr,
            body=body,
        )

from __future__ import annotations

import tokenize
from typing import List, Sequence

from ...etc import TokenStream
from ...parse import PyLogicNode
from ..build_context import build_block
from ..ir_dataclass import PyIRNode, PyIRWith, PyIRWithItem
from .statement_builder import StatementBuilder


class WithBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if not node.origins:
            raise ValueError("У PyLogicNode.WITH отсутствуют origins")

        raw = node.origins[0]
        tokens: List[tokenize.TokenInfo] = [
            t
            for t in raw.tokens
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
            return []

        first = tokens[0]
        if first.type != tokenize.NAME or first.string != "with":
            raise SyntaxError("Некорректный оператор with")

        line, col = first.start

        if tokens[-1].type != tokenize.OP or tokens[-1].string != ":":
            raise SyntaxError("Заголовок with должен заканчиваться ':'")

        header_tokens = tokens[1:-1]
        header_tokens = WithBuilder._strip_outer_parens(header_tokens)

        if not header_tokens:
            raise SyntaxError("Ожидался как минимум один элемент with")

        item_tokens_list = WithBuilder._split_top_level_commas(header_tokens)
        if any(len(part) == 0 for part in item_tokens_list):
            raise SyntaxError("Лишняя запятая или пустой элемент with не поддерживается")

        items: list[PyIRWithItem] = []
        for item_tokens in item_tokens_list:
            ctx_tokens, opt_tokens = WithBuilder._split_item_as(item_tokens)

            if not ctx_tokens:
                raise SyntaxError("Отсутствует context-выражение в элементе with")

            ctx_stream = TokenStream(ctx_tokens)
            ctx_expr = StatementBuilder._parse_expression(ctx_stream)
            if not ctx_stream.eof():
                raise SyntaxError("Некорректное context-выражение в элементе with")

            opt_vars = None
            if opt_tokens is not None:
                if not opt_tokens:
                    raise SyntaxError("Ожидалась цель после 'as' в элементе with")

                opt_stream = TokenStream(opt_tokens)
                opt_vars = StatementBuilder._parse_expression(opt_stream)
                if not opt_stream.eof():
                    raise SyntaxError("Некорректная цель 'as' в элементе with")

            iline, icol = ctx_tokens[0].start
            items.append(
                PyIRWithItem(
                    line=iline,
                    offset=icol,
                    context_expr=ctx_expr,
                    optional_vars=opt_vars,
                )
            )

        body_nodes = build_block(node.children)

        return [
            PyIRWith(
                line=line,
                offset=col,
                items=items,
                body=body_nodes,
            )
        ]

    @staticmethod
    def _strip_outer_parens(
        tokens: List[tokenize.TokenInfo],
    ) -> List[tokenize.TokenInfo]:
        if len(tokens) < 2:
            return tokens

        if not (tokens[0].type == tokenize.OP and tokens[0].string == "("):
            return tokens
        if not (tokens[-1].type == tokenize.OP and tokens[-1].string == ")"):
            return tokens

        depth = 0
        for i, t in enumerate(tokens):
            if t.type == tokenize.OP and t.string == "(":
                depth += 1
            elif t.type == tokenize.OP and t.string == ")":
                depth -= 1
                if depth < 0:
                    return tokens

            if depth == 0 and i != len(tokens) - 1:
                return tokens

        if depth != 0:
            return tokens

        return tokens[1:-1]

    @staticmethod
    def _split_top_level_commas(
        tokens: List[tokenize.TokenInfo],
    ) -> List[List[tokenize.TokenInfo]]:
        out: List[List[tokenize.TokenInfo]] = []
        acc: List[tokenize.TokenInfo] = []

        depth_paren = 0
        depth_brack = 0
        depth_brace = 0

        for t in tokens:
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
                t.type == tokenize.OP
                and t.string == ","
                and depth_paren == 0
                and depth_brack == 0
                and depth_brace == 0
            ):
                out.append(acc)
                acc = []
            else:
                acc.append(t)

        out.append(acc)
        return out

    @staticmethod
    def _split_item_as(
        item_tokens: List[tokenize.TokenInfo],
    ) -> tuple[List[tokenize.TokenInfo], List[tokenize.TokenInfo] | None]:
        depth_paren = 0
        depth_brack = 0
        depth_brace = 0

        for i, t in enumerate(item_tokens):
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
                and t.string == "as"
            ):
                left = item_tokens[:i]
                right = item_tokens[i + 1 :]
                return left, right

        return item_tokens, None

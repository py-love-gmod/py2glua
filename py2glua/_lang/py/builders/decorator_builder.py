import tokenize
from typing import Sequence

from ...etc import TokenStream
from ...parse import PyLogicNode
from ..ir_dataclass import PyIRCall, PyIRDecorator, PyIRNode
from .statement_builder import StatementBuilder


class DecoratorBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if not node.origins:
            return []

        raw = node.origins[0]

        tokens = [
            t
            for t in raw.tokens
            if isinstance(t, tokenize.TokenInfo)
            and t.type not in (tokenize.NL, tokenize.NEWLINE)
        ]

        if not tokens or tokens[0].string != "@":
            raise SyntaxError("Некорректный синтаксис декоратора")

        tokens = tokens[1:]
        if not tokens:
            raise SyntaxError("Отсутствует выражение декоратора")

        line, col = tokens[0].start

        stream = TokenStream(tokens)
        expr = StatementBuilder._parse_postfix(stream)

        if not stream.eof():
            raise SyntaxError("Некорректное выражение декоратора")

        if isinstance(expr, PyIRCall):
            dec_expr = expr.func
            args_p = expr.args_p
            args_kw = expr.args_kw
        else:
            dec_expr = expr
            args_p = []
            args_kw = {}

        return [
            PyIRDecorator(
                line=line,
                offset=col,
                exper=dec_expr,
                args_p=args_p,
                args_kw=args_kw,
            )
        ]

import tokenize
from typing import Sequence

from ...etc import TokenStream
from ...parse import PyLogicNode
from ..ir_dataclass import PyIRAttribute, PyIRCall, PyIRDecorator, PyIRNode, PyIRVarUse
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
            raise SyntaxError("Invalid decorator syntax")

        tokens = tokens[1:]
        if not tokens:
            raise SyntaxError("Decorator name is missing")

        line, col = tokens[0].start

        stream = TokenStream(tokens)
        expr = StatementBuilder._parse_postfix(stream)

        if not stream.eof():
            raise SyntaxError("Invalid decorator expression")

        if not isinstance(expr, PyIRCall):
            return [
                PyIRDecorator(
                    line=line,
                    offset=col,
                    name=DecoratorBuilder._extract_name(expr),
                    args_p=[],
                    args_kw={},
                )
            ]

        return [
            PyIRDecorator(
                line=line,
                offset=col,
                name=DecoratorBuilder._extract_name(expr.func),
                args_p=expr.args_p,
                args_kw=expr.args_kw,
            )
        ]

    @staticmethod
    def _extract_name(node: PyIRNode) -> str:
        if isinstance(node, PyIRVarUse):
            return node.name

        if isinstance(node, PyIRAttribute):
            return f"{DecoratorBuilder._extract_name(node.value)}.{node.attr}"

        raise SyntaxError("Unsupported decorator expression")

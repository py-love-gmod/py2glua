import tokenize
from typing import Sequence

from ....glua.core.types import nil
from ...etc import TokenStream
from ...parse import PyLogicNode
from ..ir_dataclass import PyIRConstant, PyIRNode, PyIRReturn
from .statement_builder import StatementBuilder


class ReturnBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        raw = node.origins[0]

        tokens = [
            t
            for t in raw.tokens
            if isinstance(t, tokenize.TokenInfo)
            and t.type not in (tokenize.NL, tokenize.NEWLINE)
        ]

        line, col = tokens[0].start

        # return
        if len(tokens) == 1:
            return [
                PyIRReturn(
                    line=line,
                    offset=col,
                    value=PyIRConstant(line=line, offset=col, value=nil),
                )
            ]

        # return expr
        stream = TokenStream(tokens[1:])
        expr = StatementBuilder._parse_expression(stream)

        if not stream.eof():
            raise SyntaxError("Invalid expression in return")

        return [
            PyIRReturn(
                line=line,
                offset=col,
                value=expr,
            )
        ]

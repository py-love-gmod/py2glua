import tokenize
from typing import Sequence

from ...etc import TokenStream
from ...parse import PyLogicNode
from ..ir_dataclass import LuaNil, PyIRConstant, PyIRNode, PyIRReturn
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
                    value=PyIRConstant(line=line, offset=col, value=LuaNil),
                )
            ]

        # return expr
        stream = TokenStream(tokens[1:])
        expr = StatementBuilder._parse_expression(stream)

        if not stream.eof():
            from ...._cli import CompilerExit

            CompilerExit.user_error(
                "Неподдерживаемое выражение в return.\n"
                "Вероятно, используется форма, которая пока не поддерживается ",
                path=None,
                line=line,
                offset=col,
                show_path=False,
            )

        return [
            PyIRReturn(
                line=line,
                offset=col,
                value=expr,
            )
        ]

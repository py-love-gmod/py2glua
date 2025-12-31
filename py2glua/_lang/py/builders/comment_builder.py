import tokenize
from typing import Sequence

from ...parse import PyLogicNode
from ...parse.py_parser import RawSyntaxNodeKind
from ..ir_dataclass import PyIRComment, PyIRNode


class CommentBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if not node.origins:
            return []

        lines = []
        line = None
        col = None

        for raw in node.origins:
            toks = [t for t in raw.tokens if isinstance(t, tokenize.TokenInfo)]
            if not toks:
                continue

            if line is None:
                line, col = toks[0].start

            if raw.kind == RawSyntaxNodeKind.DOCSTRING:  # строка без кавычек
                val = toks[0].string
                lines.append(val)

            else:  # COMMENT
                text = toks[0].string
                lines.append(text.lstrip("#").rstrip())

        return [
            PyIRComment(
                line=line,
                offset=col,
                value="\n".join(lines),
            )
        ]

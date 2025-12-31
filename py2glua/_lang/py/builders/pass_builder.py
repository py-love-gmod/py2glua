import tokenize
from typing import Sequence

from ...parse import PyLogicNode
from ..ir_dataclass import PyIRNode, PyIRPass


class PassBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        raw = node.origins[0]
        tok = next(t for t in raw.tokens if isinstance(t, tokenize.TokenInfo))
        line, col = tok.start

        return [PyIRPass(line=line, offset=col)]

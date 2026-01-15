import tokenize
from typing import Sequence

from ...parse import PyLogicKind, PyLogicNode
from ..ir_dataclass import PyIRClassDef, PyIRNode


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

        if not tokens:
            raise SyntaxError("Empty class header")

        if tokens[0].type != tokenize.NAME or tokens[0].string != "class":
            raise SyntaxError("Invalid class declaration")

        if len(tokens) < 2 or tokens[1].type != tokenize.NAME:
            raise SyntaxError("Expected class name")

        name_tok = tokens[1]
        class_name = name_tok.string

        for t in tokens:
            if t.type == tokenize.OP and t.string == "(":
                raise SyntaxError("Наследование классов не поддерживается в py2glua")

        body_children = list(node.children)

        from ..ir_builder import PyIRBuilder

        body = PyIRBuilder._build_ir_block(body_children)

        line, col = name_tok.start

        return [
            PyIRClassDef(
                line=line,
                offset=col,
                name=class_name,
                decorators=[],
                body=body,
            )
        ]

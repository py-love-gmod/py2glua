import tokenize
from typing import Sequence

from ...etc import TokenStream
from ...parse import PyLogicKind, PyLogicNode
from ..ir_builder import PyIRBuilder
from ..ir_dataclass import PyIRIf, PyIRNode
from .statement_builder import StatementBuilder


class BranchBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if node.kind is not PyLogicKind.BRANCH:
            raise ValueError("BranchBuilder expects PyLogicKind.BRANCH")

        if not node.children:
            raise SyntaxError("Empty branch node")

        parts = node.children
        if any(p.kind is not PyLogicKind.BRANCH_PART for p in parts):
            raise SyntaxError("Invalid BRANCH structure: expected BRANCH_PART children")

        built = BranchBuilder._build_chain(parts)
        return [built]

    @staticmethod
    def _build_chain(parts: list[PyLogicNode]) -> PyIRIf:
        first_hdr = BranchBuilder._get_header(parts[0])
        kw = BranchBuilder._header_keyword(first_hdr)
        if kw != "if":
            raise SyntaxError("Branch chain must start with 'if'")

        test = BranchBuilder._parse_if_test(first_hdr)
        body = PyIRBuilder._build_ir_block(parts[0].children)

        root = PyIRIf(
            line=first_hdr.tokens[0].start[0],
            offset=first_hdr.tokens[0].start[1],
            test=test,
            body=body,
            orelse=[],
        )

        cur = root

        for p in parts[1:]:
            hdr = BranchBuilder._get_header(p)
            kw = BranchBuilder._header_keyword(hdr)

            if kw == "elif":
                test2 = BranchBuilder._parse_if_test(hdr)
                body2 = PyIRBuilder._build_ir_block(p.children)
                nxt = PyIRIf(
                    line=hdr.tokens[0].start[0],
                    offset=hdr.tokens[0].start[1],
                    test=test2,
                    body=body2,
                    orelse=[],
                )
                cur.orelse = [nxt]
                cur = nxt
                continue

            if kw == "else":
                cur.orelse = PyIRBuilder._build_ir_block(p.children)
                continue

            raise SyntaxError(f"Unexpected branch header: {kw!r}")

        return root

    @staticmethod
    def _get_header(part: PyLogicNode):
        if not part.origins:
            raise ValueError("BRANCH_PART has no origins")
        return part.origins[0]

    @staticmethod
    def _header_keyword(raw_header) -> str:
        toks = [t for t in raw_header.tokens if isinstance(t, tokenize.TokenInfo)]
        if not toks or toks[0].type != tokenize.NAME:
            raise SyntaxError("Invalid branch header")

        return toks[0].string

    @staticmethod
    def _parse_if_test(raw_header) -> PyIRNode:
        toks = [
            t
            for t in raw_header.tokens
            if isinstance(t, tokenize.TokenInfo)
            and t.type
            not in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            )
        ]
        if not toks:
            raise SyntaxError("Invalid branch header")

        kw = toks[0].string
        if kw not in ("if", "elif"):
            raise SyntaxError(f"Expected if/elif, got {kw!r}")

        if toks[-1].type != tokenize.OP or toks[-1].string != ":":
            raise SyntaxError("Branch header must end with ':'")

        expr_tokens = toks[1:-1]
        if not expr_tokens:
            raise SyntaxError("Missing condition in if/elif")

        stream = TokenStream(expr_tokens)
        test = StatementBuilder._parse_expression(stream)
        if not stream.eof():
            raise SyntaxError("Invalid condition expression in if/elif")

        return test

import tokenize
from collections.abc import Sequence

from ...etc import TokenStream
from ...parse import PyLogicKind, PyLogicNode
from ..build_context import build_block
from ..ir_dataclass import PyIRIf, PyIRNode
from .statement_builder import StatementBuilder


class BranchBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if node.kind is not PyLogicKind.BRANCH:
            raise ValueError("BranchBuilder ожидает PyLogicKind.BRANCH")

        if not node.children:
            raise SyntaxError("Пустой узел ветвления")

        parts = node.children
        if any(p.kind is not PyLogicKind.BRANCH_PART for p in parts):
            raise SyntaxError(
                "Некорректная структура BRANCH: ожидались дочерние BRANCH_PART"
            )

        built = BranchBuilder._build_chain(parts)
        return [built]

    @staticmethod
    def _build_chain(parts: list[PyLogicNode]) -> PyIRIf:
        first_hdr = BranchBuilder._get_header(parts[0])
        kw = BranchBuilder._header_keyword(first_hdr)
        if kw != "if":
            raise SyntaxError("Цепочка ветвления должна начинаться с 'if'")

        test = BranchBuilder._parse_if_test(first_hdr)
        body = build_block(parts[0].children)

        first_tok = BranchBuilder._first_token(first_hdr)
        line, col = first_tok.start

        root = PyIRIf(
            line=line,
            offset=col,
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
                body2 = build_block(p.children)

                t0 = BranchBuilder._first_token(hdr)
                l2, c2 = t0.start

                nxt = PyIRIf(
                    line=l2,
                    offset=c2,
                    test=test2,
                    body=body2,
                    orelse=[],
                )
                cur.orelse = [nxt]
                cur = nxt
                continue

            if kw == "else":
                cur.orelse = build_block(p.children)
                continue

            raise SyntaxError(f"Неожиданный заголовок ветвления: {kw!r}")

        return root

    # region header helpers
    @staticmethod
    def _get_header(part: PyLogicNode):
        if not part.origins:
            raise ValueError("У BRANCH_PART отсутствуют origins")

        return part.origins[0]

    @staticmethod
    def _first_token(raw_header) -> tokenize.TokenInfo:
        toks = [t for t in raw_header.tokens if isinstance(t, tokenize.TokenInfo)]
        if not toks:
            raise SyntaxError("Пустые токены заголовка ветвления")

        return toks[0]

    @staticmethod
    def _header_keyword(raw_header) -> str:
        tok0 = BranchBuilder._first_token(raw_header)
        if tok0.type != tokenize.NAME:
            raise SyntaxError("Некорректный заголовок ветвления")

        return tok0.string

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
            raise SyntaxError("Некорректный заголовок ветвления")

        kw = toks[0].string
        if kw not in ("if", "elif"):
            raise SyntaxError(f"Ожидалось if/elif, получено {kw!r}")

        if toks[-1].type != tokenize.OP or toks[-1].string != ":":
            raise SyntaxError("Заголовок ветвления должен заканчиваться ':'")

        expr_tokens = toks[1:-1]
        if not expr_tokens:
            raise SyntaxError("Отсутствует условие в if/elif")

        stream = TokenStream(expr_tokens)
        test = StatementBuilder._parse_expression(stream)
        if not stream.eof():
            raise SyntaxError("Некорректное выражение условия в if/elif")

        return test

    # endregion

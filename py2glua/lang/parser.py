import ast
import tokenize
from enum import Enum, auto
from io import BytesIO


# region Tokens
class _TokenStream:
    def __init__(self, tokens: list[tokenize.TokenInfo]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> tokenize.TokenInfo | None:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]

        return None

    def advance(self) -> tokenize.TokenInfo | None:
        tok = self.peek(0)
        if tok is not None:
            self.pos += 1

        return tok

    def match(self, type_=None, value=None):
        tok = self.peek()
        if tok is None:
            return False

        if type_ is not None and tok.type != type_:
            return False

        if value is not None and tok.string != value:
            return False

        return True

    def eof(self) -> bool:
        return self.pos >= len(self.tokens)


# endregion


# region RawLex
class RawNodeKind(Enum):
    IMPORT = auto()
    FUNCTION = auto()
    OTHER = auto()


class _BaseRawLex:
    def __init__(
        self,
        kind: RawNodeKind,
        list_of_tokens: list[tokenize.TokenInfo],
    ) -> None:
        self.kind = kind
        self.tokens = list_of_tokens


class RawNode(_BaseRawLex):
    def __init__(self, kind, tokens):
        self.kind = kind
        self.tokens = tokens


# endregion


class Parser:
    # region tokens
    @staticmethod
    def _constract_tokens(source: str) -> _TokenStream:
        ast.parse(source)

        token_list = []
        for token in tokenize.tokenize(BytesIO(source.encode("utf-8")).readline):
            token_list.append(token)

        return _TokenStream(token_list)

    # endregion

    # region lex
    @classmethod
    def _constract_raw_lex(cls, token_stream: _TokenStream):
        nodes = []

        while not token_stream.eof():
            # region import
            if token_stream.match(value="import") or token_stream.match(value="from"):
                nodes.append(cls._build_raw_import(token_stream))
                continue
            # endregion

        return nodes

    @classmethod
    def _build_raw_import(cls, token_stream: _TokenStream):
        tokens = []

        balance = {"(": 0, ")": 0}

        while not token_stream.eof():
            tok = token_stream.advance()
            if tok is None:
                break

            tokens.append(tok)

            s = tok.string
            if s in balance:
                balance[s] += 1

            if tok.type in (tokenize.NEWLINE, tokenize.NL) and not (
                balance["("] > balance[")"]
            ):
                break

        return RawNode(RawNodeKind.IMPORT, tokens)

    # endregion

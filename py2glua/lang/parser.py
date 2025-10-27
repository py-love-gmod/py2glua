import ast
import tokenize
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

    def eof(self) -> bool:
        return self.pos >= len(self.tokens)


# endregion


# region RawLex
class _BaseRawLex:
    pass


class _RawLexStream:
    pass


# endregion


class Parser:
    @staticmethod
    def _constract_tokens(source: str) -> _TokenStream:
        ast.parse(source)

        token_list = []
        for tok in tokenize.tokenize(BytesIO(source.encode("utf-8")).readline):
            token_list.append(tok)

        return _TokenStream(token_list)

    @staticmethod
    def _constract_raw_lex(token_stream: _TokenStream) -> _RawLexStream: ...

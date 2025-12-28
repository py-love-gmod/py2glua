import tokenize
from typing import List


class TokenStream:
    def __init__(self, tokens: List[tokenize.TokenInfo]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self, value: int = 0) -> tokenize.TokenInfo | None:
        offset = self.index + value
        if offset >= len(self.tokens):
            return None

        return self.tokens[offset]

    def advance(self) -> tokenize.TokenInfo | None:
        tok = self.peek()
        if tok is not None:
            self.index += 1

        return tok

    def expect_op(self, value: str) -> tokenize.TokenInfo:
        tok = self.advance()
        if tok is None or tok.type != tokenize.OP or tok.string != value:
            raise SyntaxError(f"Expected {value!r}, got {tok!r}")

        return tok

    def eof(self) -> bool:
        return self.index >= len(self.tokens)

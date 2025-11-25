import tokenize


class TokenStream:
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

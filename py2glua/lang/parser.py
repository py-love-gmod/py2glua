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

    def eof(self) -> bool:
        return self.pos >= len(self.tokens)


# endregion


# region RawLex
class RawNodeKind(Enum):
    IMPORT = auto()

    FUNCTION = auto()
    CLASS = auto()

    IF = auto()
    ELIF = auto()
    ELSE = auto()

    TRY = auto()
    EXCEPT = auto()
    FINALLY = auto()

    DECORATORS = auto()

    DEL = auto()

    WHILE = auto()
    FOR = auto()

    WITH = auto()
    BLOCK = auto()

    OTHER = auto()


class _BaseRawLex:
    def __init__(
        self,
        kind: RawNodeKind,
        list_of_tokens: list["tokenize.TokenInfo | _BaseRawLex"],
    ) -> None:
        self.kind = kind
        self.tokens = list_of_tokens


class RawNode(_BaseRawLex):
    def __init__(self, kind, tokens):
        self.kind = kind
        self.tokens = tokens


# endregion

HEADER_KEYWORDS = {
    "import": RawNodeKind.IMPORT,
    "from": RawNodeKind.IMPORT,
    "del": RawNodeKind.DEL,
    "def": RawNodeKind.FUNCTION,
    "class": RawNodeKind.CLASS,
    "if": RawNodeKind.IF,
    "elif": RawNodeKind.ELIF,
    "else": RawNodeKind.ELSE,
    "try": RawNodeKind.TRY,
    "except": RawNodeKind.EXCEPT,
    "finally": RawNodeKind.FINALLY,
    "while": RawNodeKind.WHILE,
    "for": RawNodeKind.FOR,
    "with": RawNodeKind.WITH,
}


class Parser:
    @classmethod
    def parse(cls, source: str) -> list[RawNode]:
        stream = cls._construct_tokens(source)
        raw = cls._construct_raw_lex(stream)
        return cls._expand_blocks(raw)

    # region tokens
    @staticmethod
    def _construct_tokens(source: str) -> _TokenStream:
        ast.parse(source)

        token_list = []
        for token in tokenize.tokenize(BytesIO(source.encode("utf-8")).readline):
            if token.type == tokenize.ENCODING:
                continue

            token_list.append(token)

        return _TokenStream(token_list)

    # endregion

    # region lex
    @classmethod
    def _construct_raw_lex(cls, token_stream: _TokenStream):
        nodes = []

        while not token_stream.eof():
            tok = token_stream.peek()
            assert tok is not None
            tok_string = tok.string

            if tok_string in {"global", "nonlocal", "async", "await", "yield"}:
                raise SyntaxError(
                    "Forbidden construct in this dialect"
                )  # TODO: normal raize

            if tok_string == "@":
                nodes.append(cls._build_raw_decorator(token_stream))
                continue

            if tok_string in HEADER_KEYWORDS:
                func = getattr(cls, f"_build_raw_{tok_string}")
                nodes.append(func(token_stream))
                continue

            if tok.type == tokenize.INDENT:
                nodes.append(cls._build_raw_indent(token_stream))
                continue

            token_stream.advance()

        return nodes

    @classmethod
    def _expand_blocks(cls, nodes: list[RawNode]) -> list[RawNode]:
        out: list[RawNode] = []

        for n in nodes:
            if n.kind == RawNodeKind.BLOCK:
                toks = [t for t in n.tokens if isinstance(t, tokenize.TokenInfo)]

                if toks and toks[0].type == tokenize.INDENT:
                    toks = toks[1:]

                if toks and toks[-1].type == tokenize.DEDENT:
                    toks = toks[:-1]

                inner_stream = _TokenStream(toks)
                inner_nodes = cls._construct_raw_lex(inner_stream)

                n.tokens = cls._expand_blocks(inner_nodes)
                out.append(n)
                continue

            if any(isinstance(t, _BaseRawLex) for t in n.tokens):
                new_tokens = []
                for t in n.tokens:
                    if isinstance(t, RawNode):
                        new_tokens.extend(cls._expand_blocks([t]))

                    else:
                        new_tokens.append(t)

                n.tokens = new_tokens

            out.append(n)

        return out

    # endregion

    # region generic collectors
    @classmethod
    def _build_until(cls, token_stream, kind, stop_predicate):
        tokens = []
        balance = {"(": 0, ")": 0, "[": 0, "]": 0, "{": 0, "}": 0}

        def paren_open() -> bool:
            return (
                balance["("] > balance[")"]
                or balance["["] > balance["]"]
                or balance["{"] > balance["}"]
            )

        while not token_stream.eof():
            tok = token_stream.advance()
            if tok is None:
                break

            tokens.append(tok)

            s = tok.string
            if s in balance:
                balance[s] += 1

            if stop_predicate(tok, paren_open):
                break

        return RawNode(kind, tokens)

    @classmethod
    def _build_finish_newline(cls, token_stream, kind):
        def stop(tok: tokenize.TokenInfo, paren_open):
            if tok.type not in (tokenize.NEWLINE, tokenize.NL):
                return False

            if tok.line and tok.line.rstrip().endswith("\\"):
                return False

            return not paren_open()

        return cls._build_until(token_stream, kind, stop)

    @classmethod
    def _build_finish_colon(cls, token_stream, kind):
        def stop(tok: tokenize.TokenInfo, paren_open):
            return tok.string == ":" and not paren_open()

        return cls._build_until(token_stream, kind, stop)

    # endregion

    # region specific builders
    @classmethod
    def _build_raw_import(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_newline(token_stream, RawNodeKind.IMPORT)

    @classmethod
    def _build_raw_from(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_newline(token_stream, RawNodeKind.IMPORT)

    @classmethod
    def _build_raw_del(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_newline(token_stream, RawNodeKind.DEL)

    @classmethod
    def _build_raw_decorator(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_newline(token_stream, RawNodeKind.DECORATORS)

    @classmethod
    def _build_raw_def(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.FUNCTION)

    @classmethod
    def _build_raw_class(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.CLASS)

    @classmethod
    def _build_raw_if(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.IF)

    @classmethod
    def _build_raw_elif(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.ELIF)

    @classmethod
    def _build_raw_else(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.ELSE)

    @classmethod
    def _build_raw_try(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.TRY)

    @classmethod
    def _build_raw_except(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.EXCEPT)

    @classmethod
    def _build_raw_finally(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.FINALLY)

    @classmethod
    def _build_raw_while(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.WHILE)

    @classmethod
    def _build_raw_for(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.FOR)

    @classmethod
    def _build_raw_with(cls, token_stream: _TokenStream) -> RawNode:
        return cls._build_finish_colon(token_stream, RawNodeKind.WITH)

    @classmethod
    def _build_raw_indent(cls, token_stream: _TokenStream) -> RawNode:
        tokens: list[tokenize.TokenInfo] = []

        first = token_stream.advance()
        if first is None or first.type != tokenize.INDENT:
            raise SyntaxError("Expected INDENT at start of block")

        tokens.append(first)

        depth = 1
        while not token_stream.eof():
            tok = token_stream.advance()
            if tok is None:
                break

            tokens.append(tok)

            if tok.type == tokenize.INDENT:
                depth += 1

            elif tok.type == tokenize.DEDENT:
                depth -= 1
                if depth == 0:
                    break

        return RawNode(RawNodeKind.BLOCK, tokens)

    # endregion

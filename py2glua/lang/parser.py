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


# region RawNonTerminal
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
    RETURN = auto()
    PASS = auto()

    WHILE = auto()
    FOR = auto()
    WITH = auto()

    BLOCK = auto()
    COMMENT = auto()
    DOCSTRING = auto()
    OTHER = auto()


class _BaseRawLex:
    def __init__(
        self,
        kind: RawNodeKind,
        tokens: list["tokenize.TokenInfo | _BaseRawLex"],
    ) -> None:
        self.kind = kind
        self.tokens = tokens


class RawNode(_BaseRawLex):
    def __init__(self, kind: RawNodeKind, tokens: list):
        self.kind = kind
        self.tokens: list = tokens


# endregion


_HEADER_KEYWORDS = {
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
    "return": RawNodeKind.RETURN,
    "pass": RawNodeKind.PASS,
}


class Parser:
    @classmethod
    def parse(cls, source: str) -> list[RawNode]:
        stream = cls._construct_tokens(source)
        raw = cls._construct_raw_non_terminal(stream)
        expanded = cls._expand_blocks(raw)
        cls._promote_leading_docstring(expanded)
        return expanded

    # region tokens
    @staticmethod
    def _construct_tokens(source: str) -> _TokenStream:
        try:
            compile(source, "<py2glua-validate>", "exec")

        except SyntaxError as e:
            raise SyntaxError(
                f"Invalid Python syntax: {e.msg} (line {e.lineno}, offset {e.offset})"
            )

        token_list: list[tokenize.TokenInfo] = []
        for token in tokenize.tokenize(BytesIO(source.encode("utf-8")).readline):
            if token.type == tokenize.ENCODING:
                continue

            token_list.append(token)

        return _TokenStream(token_list)

    # endregion

    # region Non Terminal
    @classmethod
    def _construct_raw_non_terminal(cls, token_stream: _TokenStream):
        nodes: list[RawNode] = []

        while not token_stream.eof():
            tok = token_stream.peek()
            if tok is None:
                break

            tok_string = tok.string
            tok_type = tok.type

            if tok_type == tokenize.COMMENT:
                nodes.append(cls._build_raw_comment(token_stream))
                continue

            if tok_type in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                token_stream.advance()
                continue

            if tok_string in {"global", "nonlocal", "async", "await", "yield"}:
                raise SyntaxError(
                    f"global, nonlocal, async, await, yield keywords are not supported in py2glua\n"
                    f"LINE|OFFSET: {tok.start[0]}|{tok.start[1]}"
                )

            if tok_string == "@":
                nodes.extend(cls._build_raw_decorator(token_stream))
                continue

            if tok_string in _HEADER_KEYWORDS:
                func = getattr(cls, f"_build_raw_{tok_string}")
                res = func(token_stream)
                if isinstance(res, tuple):
                    nodes.extend(res)

                elif isinstance(res, list):
                    nodes.extend(res)

                else:
                    nodes.append(res)

                continue

            if tok_type == tokenize.INDENT:
                nodes.append(cls._build_raw_indent(token_stream))
                continue

            res = cls._build_raw_other(token_stream)
            if isinstance(res, list):
                nodes.extend(res)

            else:
                nodes.append(res)

        return nodes

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
        tokens: list[tokenize.TokenInfo] = []
        balance = {"(": 0, ")": 0, "[": 0, "]": 0, "{": 0, "}": 0}

        def paren_open() -> bool:
            return (
                balance["("] > balance[")"]
                or balance["["] > balance["]"]
                or balance["{"] > balance["}"]
            )

        while not token_stream.eof():
            tok = token_stream.peek()
            if tok is None:
                break

            s = tok.string
            if s in balance:
                balance[s] += 1

            if tok.type == tokenize.COMMENT and not paren_open():
                break

            if tok.type in (tokenize.NEWLINE, tokenize.NL) and not paren_open():
                tokens.append(token_stream.advance())
                break

            tokens.append(token_stream.advance())

        node = RawNode(kind, tokens)

        nxt = token_stream.peek()
        if nxt and nxt.type == tokenize.COMMENT:
            comment = cls._build_raw_comment(token_stream)
            return [node, comment]

        return [node]

    @classmethod
    def _build_finish_colon(cls, token_stream, kind):
        def stop(tok: tokenize.TokenInfo, paren_open):
            return tok.string == ":" and not paren_open()

        return cls._build_until(token_stream, kind, stop)

    @classmethod
    def _build_header_with_optional_inline_block(
        cls, token_stream: _TokenStream, kind: RawNodeKind
    ):
        header = cls._build_finish_colon(token_stream, kind)
        nxt = token_stream.peek()
        if nxt is None or nxt.type in (tokenize.NEWLINE, tokenize.NL):
            return header

        inline_stmt = cls._build_finish_newline(token_stream, RawNodeKind.OTHER)
        block = RawNode(
            RawNodeKind.BLOCK,
            [inline_stmt] if not isinstance(inline_stmt, list) else inline_stmt,
        )
        return (header, block)

    # endregion

    # region specific builders
    @classmethod
    def _build_raw_import(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawNodeKind.IMPORT)

    @classmethod
    def _build_raw_from(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawNodeKind.IMPORT)

    @classmethod
    def _build_raw_del(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawNodeKind.DEL)

    @classmethod
    def _build_raw_return(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawNodeKind.RETURN)

    @classmethod
    def _build_raw_pass(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawNodeKind.PASS)

    @classmethod
    def _build_raw_decorator(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawNodeKind.DECORATORS)

    @classmethod
    def _build_raw_def(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.FUNCTION
        )

    @classmethod
    def _build_raw_class(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.CLASS
        )

    @classmethod
    def _build_raw_if(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.IF
        )

    @classmethod
    def _build_raw_elif(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.ELIF
        )

    @classmethod
    def _build_raw_else(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.ELSE
        )

    @classmethod
    def _build_raw_try(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.TRY
        )

    @classmethod
    def _build_raw_except(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.EXCEPT
        )

    @classmethod
    def _build_raw_finally(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.FINALLY
        )

    @classmethod
    def _build_raw_while(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.WHILE
        )

    @classmethod
    def _build_raw_for(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.FOR
        )

    @classmethod
    def _build_raw_with(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawNodeKind.WITH
        )

    @classmethod
    def _build_raw_other(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawNodeKind.OTHER)

    @classmethod
    def _build_raw_indent(cls, token_stream):
        tokens: list[tokenize.TokenInfo] = []
        first = token_stream.advance()
        if first is None or first.type != tokenize.INDENT:
            raise SyntaxError("Expected INDENT at start of block")

        tokens.append(first)
        depth = 1
        while not token_stream.eof():
            look = token_stream.peek()
            if look is None:
                break

            if (
                look.type == tokenize.COMMENT
                and depth == 1
                and getattr(look, "start", (0, 1))[1] == 0
            ):
                break

            tok = token_stream.advance()
            tokens.append(tok)

            if tok.type == tokenize.INDENT:
                depth += 1

            elif tok.type == tokenize.DEDENT:
                depth -= 1
                if depth == 0:
                    break

        return RawNode(RawNodeKind.BLOCK, tokens)

    @classmethod
    def _build_raw_comment(cls, token_stream: _TokenStream) -> RawNode:
        tokens = []
        while not token_stream.eof():
            tok = token_stream.advance()
            if tok is None:
                break

            tokens.append(tok)
            if tok.type in (tokenize.NEWLINE, tokenize.NL):
                break

        return RawNode(RawNodeKind.COMMENT, tokens)

    # endregion

    # region expand & docstring
    @classmethod
    def _expand_blocks(cls, nodes: list[RawNode]) -> list[RawNode]:
        out: list[RawNode] = []
        for n in nodes:
            if n.kind == RawNodeKind.BLOCK:
                raw_children = [t for t in n.tokens if isinstance(t, RawNode)]
                if raw_children:
                    n.tokens = cls._expand_blocks(raw_children)
                    cls._promote_leading_docstring(n.tokens)
                    out.append(n)
                    continue

                toks = [t for t in n.tokens if isinstance(t, tokenize.TokenInfo)]
                if toks and toks[0].type == tokenize.INDENT:
                    toks = toks[1:]

                if toks and toks[-1].type == tokenize.DEDENT:
                    toks = toks[:-1]

                inner_stream = _TokenStream(toks)
                inner_nodes = cls._construct_raw_non_terminal(inner_stream)
                n.tokens = cls._expand_blocks(inner_nodes)
                cls._promote_leading_docstring(n.tokens)
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

    @staticmethod
    def _node_is_string_only_stmt(node: RawNode) -> bool:
        toks: list[tokenize.TokenInfo] = [
            t
            for t in node.tokens
            if isinstance(t, tokenize.TokenInfo)
            and t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT)
        ]
        return len(toks) == 1 and toks[0].type == tokenize.STRING

    @classmethod
    def _promote_leading_docstring(cls, nodes: list[RawNode]) -> None:
        i = 0
        while i < len(nodes) and nodes[i].kind == RawNodeKind.COMMENT:
            i += 1

        if i < len(nodes) and nodes[i].kind == RawNodeKind.OTHER:
            if cls._node_is_string_only_stmt(nodes[i]):
                nodes[i].kind = RawNodeKind.DOCSTRING

    # endregion

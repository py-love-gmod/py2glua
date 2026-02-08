import tokenize
from enum import Enum, auto
from io import BytesIO

from ..etc import TokenStream


# region RawSyntaxNode
class RawSyntaxNodeKind(Enum):
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


class _RawSyntaxNode:
    def __init__(
        self,
        kind: RawSyntaxNodeKind,
        tokens: list["tokenize.TokenInfo | _RawSyntaxNode"],
    ) -> None:
        self.kind = kind
        self.tokens = tokens


class RawSyntaxNode(_RawSyntaxNode):
    def __init__(self, kind: RawSyntaxNodeKind, tokens: list):
        self.kind = kind
        self.tokens: list = tokens


# endregion


_HEADER_KEYWORDS = {
    "import": RawSyntaxNodeKind.IMPORT,
    "from": RawSyntaxNodeKind.IMPORT,
    "del": RawSyntaxNodeKind.DEL,
    "def": RawSyntaxNodeKind.FUNCTION,
    "class": RawSyntaxNodeKind.CLASS,
    "if": RawSyntaxNodeKind.IF,
    "elif": RawSyntaxNodeKind.ELIF,
    "else": RawSyntaxNodeKind.ELSE,
    "try": RawSyntaxNodeKind.TRY,
    "except": RawSyntaxNodeKind.EXCEPT,
    "finally": RawSyntaxNodeKind.FINALLY,
    "while": RawSyntaxNodeKind.WHILE,
    "for": RawSyntaxNodeKind.FOR,
    "with": RawSyntaxNodeKind.WITH,
    "return": RawSyntaxNodeKind.RETURN,
    "pass": RawSyntaxNodeKind.PASS,
}


class PyParser:
    _CHAIN_CONTINUATIONS = {"elif", "else", "except", "finally"}

    @classmethod
    def parse(cls, source: str) -> list[RawSyntaxNode]:
        stream = cls._construct_tokens(source)
        raw = cls._construct_raw_non_terminal(stream)
        expanded = cls._expand_blocks(raw)
        cls._promote_leading_docstring(expanded)
        return expanded

    # region tokens
    @staticmethod
    def _construct_tokens(source: str) -> TokenStream:
        # Поддержка файлов в UTF-8 с BOM.
        if source.startswith("\ufeff"):
            source = source[1:]

        try:
            compile(source, "<py2glua-validate>", "exec")

        except SyntaxError as e:
            raise SyntaxError(
                f"Некорректный синтаксис Python: {e.msg} (line {e.lineno}, offset {e.offset})"
            )

        token_list: list[tokenize.TokenInfo] = []
        for token in tokenize.tokenize(BytesIO(source.encode("utf-8")).readline):
            if token.type == tokenize.ENCODING:
                continue
            token_list.append(token)

        return TokenStream(token_list)

    # endregion

    # region Non Terminal
    @classmethod
    def _peek_next_significant(
        cls, token_stream: TokenStream
    ) -> tokenize.TokenInfo | None:
        """
        Peek ahead skipping NL/NEWLINE/DEDENT.
        Does NOT skip INDENT: INDENT is significant for us.
        """
        j = 0
        while True:
            t = token_stream.peek(j)
            if t is None:
                return None
            if t.type in (tokenize.NL, tokenize.NEWLINE, tokenize.DEDENT):
                j += 1
                continue
            return t

    @classmethod
    def _construct_raw_non_terminal(cls, token_stream: TokenStream):
        nodes: list[RawSyntaxNode] = []
        awaiting_block = False
        pending_leading: list[RawSyntaxNode] = []

        chain_gap_comments: list[RawSyntaxNode] = []

        def attach_pending_to_block(block: RawSyntaxNode) -> None:
            nonlocal pending_leading
            if pending_leading:
                setattr(block, "_leading", list(pending_leading))
                pending_leading.clear()

        while not token_stream.eof():
            tok = token_stream.peek()
            if tok is None:
                break

            tok_string = tok.string
            tok_type = tok.type

            # comments
            if tok_type == tokenize.COMMENT:
                comment = cls._build_raw_comment(token_stream)

                if awaiting_block:
                    pending_leading.append(comment)
                    continue

                nxt = cls._peek_next_significant(token_stream)
                if (
                    nxt is not None
                    and nxt.type == tokenize.NAME
                    and nxt.string in cls._CHAIN_CONTINUATIONS
                ):
                    chain_gap_comments.append(comment)
                else:
                    nodes.append(comment)

                continue

            # trivia
            if tok_type in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                token_stream.advance()
                continue

            # unsupported keywords
            if tok_string in {
                "global",
                "nonlocal",
                "async",
                "await",
                "yield",
                "lambda",
            }:
                raise SyntaxError(
                    f"{tok_string} как ключевое слово не поддерживается\n"
                    f"LINE|OFFSET: {tok.start[0]}|{tok.start[1]}"
                )

            # decorator line(s)
            if tok_string == "@":
                if chain_gap_comments:
                    nodes.extend(chain_gap_comments)
                    chain_gap_comments.clear()

                nodes.extend(cls._build_raw_decorator(token_stream))
                awaiting_block = False
                pending_leading.clear()
                continue

            # headers
            if tok_string in _HEADER_KEYWORDS:
                if tok_string in cls._CHAIN_CONTINUATIONS and chain_gap_comments:
                    pending_leading.extend(chain_gap_comments)
                    chain_gap_comments.clear()

                if tok_string not in cls._CHAIN_CONTINUATIONS and chain_gap_comments:
                    nodes.extend(chain_gap_comments)
                    chain_gap_comments.clear()

                func = getattr(cls, f"_build_raw_{tok_string}")
                res = func(token_stream)

                if isinstance(res, tuple):
                    header, block = res
                    attach_pending_to_block(block)

                    nodes.append(header)
                    nodes.append(block)

                    awaiting_block = False
                    pending_leading.clear()

                elif isinstance(res, list):
                    nodes.extend(res)
                    awaiting_block = False
                    pending_leading.clear()

                else:
                    nodes.append(res)
                    awaiting_block = True

                continue

            if tok_type == tokenize.INDENT:
                if chain_gap_comments:
                    nodes.extend(chain_gap_comments)
                    chain_gap_comments.clear()

                block = cls._build_raw_indent(token_stream)

                if awaiting_block:
                    attach_pending_to_block(block)

                awaiting_block = False
                pending_leading.clear()
                nodes.append(block)
                continue

            if chain_gap_comments:
                nodes.extend(chain_gap_comments)
                chain_gap_comments.clear()

            res = cls._build_raw_other(token_stream)
            if isinstance(res, list):
                nodes.extend(res)

            else:
                nodes.append(res)

        if chain_gap_comments:
            nodes.extend(chain_gap_comments)
            chain_gap_comments.clear()

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

        return RawSyntaxNode(kind, tokens)

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

        node = RawSyntaxNode(kind, tokens)

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
        cls, token_stream: TokenStream, kind: RawSyntaxNodeKind
    ):
        header = cls._build_finish_colon(token_stream, kind)
        nxt = token_stream.peek()

        # IMPORTANT:
        # If the next token is COMMENT, this is still a normal suite header:
        #   if cond:  # comment
        #       ...
        # Treat comment like end-of-line, not inline suite.
        if nxt is None or nxt.type in (tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT):
            return header

        inline_tokens = []
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

            if tok.string in balance:
                balance[tok.string] += 1

            advanced = token_stream.advance()
            if advanced is None:
                break
            inline_tokens.append(advanced)

            if tok.type in (tokenize.NEWLINE, tokenize.NL) and not paren_open():
                break

        first_real: tokenize.TokenInfo | None = None
        for t in inline_tokens:
            if t.type not in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            ):
                first_real = t
                break

        if first_real is None:
            block = RawSyntaxNode(
                RawSyntaxNodeKind.BLOCK,
                [RawSyntaxNode(RawSyntaxNodeKind.OTHER, inline_tokens)],
            )
            return (header, block)

        kw = first_real.string

        if kw in _HEADER_KEYWORDS:
            build_func = getattr(cls, f"_build_raw_{kw}")
            res = build_func(TokenStream(inline_tokens))

            if isinstance(res, list):
                node = res[0]
            else:
                node = res

            block = RawSyntaxNode(RawSyntaxNodeKind.BLOCK, [node])
            return (header, block)

        node = RawSyntaxNode(RawSyntaxNodeKind.OTHER, inline_tokens)
        block = RawSyntaxNode(RawSyntaxNodeKind.BLOCK, [node])
        return (header, block)

    # endregion

    # region specific builders
    @classmethod
    def _build_raw_import(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawSyntaxNodeKind.IMPORT)

    @classmethod
    def _build_raw_from(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawSyntaxNodeKind.IMPORT)

    @classmethod
    def _build_raw_del(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawSyntaxNodeKind.DEL)

    @classmethod
    def _build_raw_return(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawSyntaxNodeKind.RETURN)

    @classmethod
    def _build_raw_pass(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawSyntaxNodeKind.PASS)

    @classmethod
    def _build_raw_decorator(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawSyntaxNodeKind.DECORATORS)

    @classmethod
    def _build_raw_def(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.FUNCTION
        )

    @classmethod
    def _build_raw_class(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.CLASS
        )

    @classmethod
    def _build_raw_if(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.IF
        )

    @classmethod
    def _build_raw_elif(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.ELIF
        )

    @classmethod
    def _build_raw_else(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.ELSE
        )

    @classmethod
    def _build_raw_try(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.TRY
        )

    @classmethod
    def _build_raw_except(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.EXCEPT
        )

    @classmethod
    def _build_raw_finally(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.FINALLY
        )

    @classmethod
    def _build_raw_while(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.WHILE
        )

    @classmethod
    def _build_raw_for(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.FOR
        )

    @classmethod
    def _build_raw_with(cls, token_stream):
        return cls._build_header_with_optional_inline_block(
            token_stream, RawSyntaxNodeKind.WITH
        )

    @classmethod
    def _build_raw_other(cls, token_stream):
        return cls._build_finish_newline(token_stream, RawSyntaxNodeKind.OTHER)

    @classmethod
    def _build_raw_indent(cls, token_stream):
        tokens: list[tokenize.TokenInfo] = []

        first = token_stream.advance()
        if first is None or first.type != tokenize.INDENT:
            raise SyntaxError("Ожидался INDENT в начале блока")

        tokens.append(first)
        depth = 1
        balance = {"(": 0, ")": 0, "[": 0, "]": 0, "{": 0, "}": 0}

        def paren_open() -> bool:
            return (
                balance["("] > balance[")"]
                or balance["["] > balance["]"]
                or balance["{"] > balance["}"]
            )

        def next_significant_type(start_offset: int) -> int | None:
            j = start_offset
            while True:
                t = token_stream.peek(j)
                if t is None:
                    return None

                if t.type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
                    j += 1
                    continue

                return t.type

        while not token_stream.eof():
            look = token_stream.peek()
            if look is None:
                break

            if look.type == tokenize.COMMENT and depth == 1 and not paren_open():
                nxt = next_significant_type(1)
                if nxt == tokenize.DEDENT:
                    break

            tok = token_stream.advance()
            tokens.append(tok)

            s = tok.string
            if s in balance:
                balance[s] += 1

            if tok.type == tokenize.INDENT:
                depth += 1
            elif tok.type == tokenize.DEDENT:
                depth -= 1
                if depth == 0:
                    break

        return RawSyntaxNode(RawSyntaxNodeKind.BLOCK, tokens)

    @classmethod
    def _build_raw_comment(cls, token_stream: TokenStream) -> RawSyntaxNode:
        tokens = []
        while not token_stream.eof():
            tok = token_stream.advance()
            if tok is None:
                break

            tokens.append(tok)
            if tok.type in (tokenize.NEWLINE, tokenize.NL):
                break

        return RawSyntaxNode(RawSyntaxNodeKind.COMMENT, tokens)

    # endregion

    # region expand & docstring
    @classmethod
    def _expand_blocks(cls, nodes: list[RawSyntaxNode]) -> list[RawSyntaxNode]:
        out: list[RawSyntaxNode] = []
        for n in nodes:
            if n.kind == RawSyntaxNodeKind.BLOCK:
                raw_children = [t for t in n.tokens if isinstance(t, RawSyntaxNode)]
                if raw_children:
                    n.tokens = cls._expand_blocks(raw_children)
                    cls._promote_leading_docstring(n.tokens)
                    lead = getattr(n, "_leading", None)
                    if lead:
                        n.tokens = lead + n.tokens

                    out.append(n)
                    continue

                toks = [t for t in n.tokens if isinstance(t, tokenize.TokenInfo)]
                if toks and toks[0].type == tokenize.INDENT:
                    toks = toks[1:]

                if toks and toks[-1].type == tokenize.DEDENT:
                    toks = toks[:-1]

                inner_stream = TokenStream(toks)
                inner_nodes = cls._construct_raw_non_terminal(inner_stream)
                n.tokens = cls._expand_blocks(inner_nodes)
                cls._promote_leading_docstring(n.tokens)
                lead = getattr(n, "_leading", None)
                if lead:
                    n.tokens = lead + n.tokens

                out.append(n)
                continue

            if any(isinstance(t, _RawSyntaxNode) for t in n.tokens):
                new_tokens = []
                for t in n.tokens:
                    if isinstance(t, RawSyntaxNode):
                        new_tokens.extend(cls._expand_blocks([t]))
                    else:
                        new_tokens.append(t)
                n.tokens = new_tokens

            out.append(n)

        return out

    @staticmethod
    def _node_is_string_only_stmt(node: RawSyntaxNode) -> bool:
        toks: list[tokenize.TokenInfo] = [
            t
            for t in node.tokens
            if isinstance(t, tokenize.TokenInfo)
            and t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT)
        ]
        return len(toks) == 1 and toks[0].type == tokenize.STRING

    @classmethod
    def _promote_leading_docstring(cls, nodes: list[RawSyntaxNode]) -> None:
        i = 0
        while i < len(nodes) and nodes[i].kind == RawSyntaxNodeKind.COMMENT:
            i += 1

        if i < len(nodes) and nodes[i].kind == RawSyntaxNodeKind.OTHER:
            if cls._node_is_string_only_stmt(nodes[i]):
                nodes[i].kind = RawSyntaxNodeKind.DOCSTRING

    # endregion

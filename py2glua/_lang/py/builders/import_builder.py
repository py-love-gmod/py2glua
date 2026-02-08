import tokenize
from typing import Sequence

from ...etc import TokenStream
from ...parse import PyLogicNode
from ..ir_dataclass import PyIRImport, PyIRImportType, PyIRNode


class ImportBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if not node.origins:
            raise ValueError("У PyLogicNode.IMPORT отсутствуют origins")

        raw = node.origins[0]
        tokens = [
            t
            for t in raw.tokens
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
            return []

        line, col = tokens[0].start
        stream = TokenStream(tokens)

        first = stream.advance()
        assert first is not None  # По факту должно быть анричебл
        if first.type != tokenize.NAME:
            raise SyntaxError("Некорректный оператор import")

        if first.string == "import":
            return ImportBuilder._build_import(stream, line, col)

        if first.string == "from":
            return [ImportBuilder._build_from_import(stream, line, col)]

        raise SyntaxError("Некорректный оператор import")

    @staticmethod
    def _build_import(stream: TokenStream, line: int, col: int) -> list[PyIRImport]:
        out: list[PyIRImport] = []

        while True:
            parts = ImportBuilder._parse_dotted_name(stream)
            if not parts:
                raise SyntaxError("Ожидалось имя модуля в import")

            alias = None
            tok = stream.peek()
            if tok and tok.type == tokenize.NAME and tok.string == "as":
                stream.advance()
                a = stream.advance()
                if a is None or a.type != tokenize.NAME:
                    raise SyntaxError("Ожидалось имя алиаса после 'as'")

                alias = a.string

            # import a.b.c as d
            if alias:
                modules = parts[:-1]
                names_out: list[str | tuple[str, str]] = [(parts[-1], alias)]

            else:
                modules = parts
                names_out = []

            out.append(
                PyIRImport(
                    line=line,
                    offset=col,
                    modules=modules,
                    names=names_out,
                    if_from=False,
                    level=0,
                    itype=PyIRImportType.UNKNOWN,
                )
            )

            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string == ",":
                stream.advance()
                continue

            break

        if not stream.eof():
            raise SyntaxError("Лишние токены в операторе import")

        return out

    @staticmethod
    def _build_from_import(stream: TokenStream, line: int, col: int) -> PyIRImport:
        level = 0
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string == ".":
                stream.advance()
                level += 1
                continue
            break

        modules: list[str] = []
        tok = stream.peek()
        if tok and tok.type == tokenize.NAME:
            modules = ImportBuilder._parse_dotted_name(stream)

        imp = stream.advance()
        if imp is None or imp.type != tokenize.NAME or imp.string != "import":
            raise SyntaxError("Ожидался 'import' в конструкции from-import")

        names: list[str | tuple[str, str]] = []

        while True:
            tok = stream.advance()
            if tok is None:
                raise SyntaxError("Неожиданный конец конструкции from-import")

            if tok.type == tokenize.OP and tok.string == "*":
                raise SyntaxError("Импорты через '*' не поддерживаются в py2glua")

            if tok.type != tokenize.NAME:
                raise SyntaxError("Ожидалось имя в конструкции from-import")

            name = tok.string
            alias = None

            nxt = stream.peek()
            if nxt and nxt.type == tokenize.NAME and nxt.string == "as":
                stream.advance()
                a = stream.advance()
                if a is None or a.type != tokenize.NAME:
                    raise SyntaxError("Ожидался алиас после 'as'")

                alias = a.string

            names.append((name, alias) if alias else name)

            nxt = stream.peek()
            if nxt and nxt.type == tokenize.OP and nxt.string == ",":
                stream.advance()
                continue

            break

        if not stream.eof():
            raise SyntaxError("Лишние токены в конструкции from-import")

        return PyIRImport(
            line=line,
            offset=col,
            modules=modules,
            names=names,
            if_from=True,
            level=level,
            itype=PyIRImportType.UNKNOWN,
        )

    @staticmethod
    def _parse_dotted_name(stream: TokenStream) -> list[str]:
        parts: list[str] = []

        tok = stream.advance()
        if tok is None or tok.type != tokenize.NAME:
            return []

        parts.append(tok.string)

        while True:
            dot = stream.peek()
            if not (dot and dot.type == tokenize.OP and dot.string == "."):
                break

            stream.advance()
            nxt = stream.advance()
            if nxt is None or nxt.type != tokenize.NAME:
                raise SyntaxError("Ожидалось имя после '.'")

            parts.append(nxt.string)

        return parts

from enum import IntEnum, auto
from pathlib import Path
from typing import Any


class TokenType(IntEnum):
    INDENT = auto()
    DEDENT = auto()

    KEY_WORD = auto()

    NAME = auto()

    DOT = auto()  # .
    COMMA = auto()  # ,
    SEMICOLON = auto()  # ;
    ARROW = auto()  # ->
    AT = auto()  # @

    OPERATOR = auto()

    LEFT_PARENTESI = auto()  # (
    RIGHT_PARENTESI = auto()  # )

    LEFT_S_PARENTESI = auto()  # [
    RIGHT_S_PARENTESI = auto()  # ]

    LEFT_F_PARENTESI = auto()  # {
    RIGHT_F_PARENTESI = auto()  # }

    NUMBER = auto()

    STRING = auto()

    COMMENT = auto()  # any comment any line long

    NEW_LINE = auto()
    EOF = auto()


class FileReader:
    """Самая базовая санитизация файла который пришёл на разбор"""

    @classmethod
    def read(cls, path: Path) -> list[Any]:
        source = path.read_text(encoding="utf-8-sig")

        try:
            compile(source, str(path), "exec")

        except SyntaxError as e:
            raise SyntaxError(
                f"Некорректный синтаксис Python: {e.msg} (line {e.lineno}, offset {e.offset})"
            ) from None

        return cls._parce_source(source)

    @classmethod
    def _parce_source(cls, source: str) -> list[Any]: ...

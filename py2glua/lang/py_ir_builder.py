from dataclasses import dataclass
from pathlib import Path

from .parser import Parser, RawNode, RawNodeKind
from .py_ir import PyIRFile, PyIRNode


class PyIRBuilder:
    @classmethod
    def build_ir(cls, path_to_file: Path):
        list_of_raw_lex = Parser.parse(path_to_file.read_text(encoding="utf-8-sig"))

import tokenize
from typing import Sequence

from .py_ir_dataclass import PyIRFile, PyIRNode


class StatementCompiler:
    @staticmethod
    def compile_line(
        tokens: list[tokenize.TokenInfo],
        file_obj: PyIRFile,
    ) -> Sequence[PyIRNode]:
        pass

    @staticmethod
    def compile_expres(
        tokens: list[tokenize.TokenInfo],
        file_obj: PyIRFile,
    ) -> PyIRNode:
        pass

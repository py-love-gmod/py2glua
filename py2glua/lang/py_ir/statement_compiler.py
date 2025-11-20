import tokenize

from ..python_etc import TokenStream
from .py_ir_dataclass import PyIRNode


class StatementCompiler:
    @staticmethod
    def compile_expres(
        tokens: list[tokenize.TokenInfo],
        parant_obj: PyIRNode,
    ) -> PyIRNode: ...

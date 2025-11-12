import tokenize

from .py_ir_dataclass import PyIRNode


class StatementCompiler:
    @staticmethod
    def compile_line(
        tokens: list[tokenize.TokenInfo],
        parant_obj: PyIRNode,
    ) -> list[PyIRNode]:
        pass

    @staticmethod
    def compile_expres(
        tokens: list[tokenize.TokenInfo],
        parant_obj: PyIRNode,
    ) -> PyIRNode:
        pass

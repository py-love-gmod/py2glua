import ast

from .convertor import FunctionEmitter


class Compiler:
    def __init__(self):
        self.func_emitter = FunctionEmitter()

    def compile_str(self, code: str) -> str:
        tree = ast.parse(code)
        out = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                out.append(self.func_emitter.emit(node))

        return "\n\n".join(out)

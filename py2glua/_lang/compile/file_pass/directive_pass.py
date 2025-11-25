from ...py.py_ir_dataclass import (
    PyIRConstant,
    PyIRFile,
    PyIRFunctionDef,
)
from ..compile_exception import CompileError
from ..compiler import Compiler


class DirectivePass:
    @staticmethod
    def run(file: PyIRFile, compiler: Compiler):
        for node in file.body:
            DirectivePass._process_node(node)

    @staticmethod
    def _process_node(node):
        if isinstance(node, PyIRFunctionDef):
            DirectivePass._process_function(node)

    @staticmethod
    def _process_function(fn: PyIRFunctionDef):
        to_remove = []

        for dec in fn.decorators:
            name = dec.name

            if name == "CompilerDirective.debug_only":
                fn.context.meta["debug_only"] = True
                to_remove.append(dec)

            elif name == "CompilerDirective.lazy_compile":
                fn.context.meta["lazy"] = True
                to_remove.append(dec)

            elif name == "CompilerDirective.inline":
                fn.context.meta["inline"] = True
                to_remove.append(dec)

            elif name == "CompilerDirective.gmod_api":
                if not dec.args_p:
                    raise CompileError("gmod_api decorator requires 1 argument")

                arg = dec.args_p[0]
                if not isinstance(arg, PyIRConstant) or not isinstance(arg.value, str):
                    raise CompileError("gmod_api argument must be a string literal")

                api_name = arg.value
                fn.context.meta["gmod_api"] = api_name
                fn.name = api_name

                to_remove.append(dec)

        for r in to_remove:
            fn.decorators.remove(r)

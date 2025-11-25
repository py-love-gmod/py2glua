from __future__ import annotations

from ...py.py_ir_dataclass import (
    PyIRAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRContext,
    PyIRFile,
    PyIRFunctionDef,
)
from ..compile_exception import CompileError
from ..compiler import Compiler


class GlobalDirectivePass:
    @staticmethod
    def run(file: PyIRFile, compiler: Compiler):
        ctx: PyIRContext = file.context
        if "globals" not in ctx.meta:
            ctx.meta["globals"] = {}

        for node in file.body:
            GlobalDirectivePass._process_node(node, ctx)

    @staticmethod
    def _process_node(node, ctx: PyIRContext):
        if isinstance(node, (PyIRFunctionDef, PyIRClassDef)):
            GlobalDirectivePass._process_mark(node, ctx)

        if isinstance(node, PyIRAssign):
            GlobalDirectivePass._process_var_assign(node, ctx)

    @staticmethod
    def _process_mark(node: PyIRFunctionDef | PyIRClassDef, ctx: PyIRContext):
        to_remove = []
        for dec in node.decorators:
            if dec.name == "Global.mark":
                external = False
                if "external" in dec.args_kw:
                    arg = dec.args_kw["external"]
                    if isinstance(arg, PyIRConstant):
                        external = bool(arg.value)
                    else:
                        raise CompileError("@Global.mark(external=...) must be literal")

                ctx.meta["globals"][node.name] = {"external": external}
                to_remove.append(dec)

        for d in to_remove:
            node.decorators.remove(d)

    @staticmethod
    def _process_var_assign(node: PyIRAssign, ctx: PyIRContext):
        if len(node.targets) != 1:
            return

        target = node.targets[0]
        if not hasattr(target, "name"):
            return

        name = target.name  # pyright: ignore[reportAttributeAccessIssue]
        value = node.value

        if isinstance(value, PyIRCall):
            func = value.func  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(func, "attr") and func.attr == "var":
                if hasattr(func.value, "name") and func.value.name == "Global":
                    external = False
                    if "external" in value.args_kw:
                        arg = value.args_kw["external"]
                        if isinstance(arg, PyIRConstant):
                            external = bool(arg.value)

                        else:
                            raise CompileError(
                                "Global.var(external=...) must be literal"
                            )

                    ctx.meta["globals"][name] = {"external": external}

                    if value.args_p:
                        node.value = value.args_p[0]

                    else:
                        node.value = PyIRConstant(0, 0, None)

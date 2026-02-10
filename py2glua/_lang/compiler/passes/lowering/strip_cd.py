from __future__ import annotations

from ....compiler.passes.analysis.symlinks import SymLinkContext
from ....py.ir_dataclass import PyIRClassDef, PyIRFile, PyIRNode
from ..common import GLUA_INTERNAL_NAMESPACE_PREFIX


class StripCompilerDirectiveDefPass:
    """
    Removes the *definition* of class CompilerDirective from INTERNAL output,
    without relying on @no_compile on itself (avoids weird self-references).

    It only triggers for internal modules.
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if ir.path is None:
            return ir

        try:
            ctx.ensure_module_index()

        except Exception:
            return ir

        module_name = None
        try:
            module_name = ctx.module_name_by_path.get(ir.path)

        except Exception:
            module_name = None

        if not module_name or not module_name.startswith(
            GLUA_INTERNAL_NAMESPACE_PREFIX
        ):
            return ir

        def keep(n: PyIRNode) -> bool:
            return not (isinstance(n, PyIRClassDef) and n.name == "CompilerDirective")

        ir.body = [n for n in ir.body if keep(n)]
        return ir

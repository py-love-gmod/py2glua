from __future__ import annotations

from .....config import Py2GluaConfig
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRConstant,
    PyIRFile,
    PyIRImport,
    PyIRNode,
)
from ..rewrite_utils import rewrite_expr_with_store_default
from ..analysis.symlinks import PyIRSymLink, SymLinkContext
from ..common import (
    collect_core_compiler_directive_local_symbol_ids,
    iter_imported_names,
)


class FoldCompileTimeBoolConstsPass:
    TYPING_MODULE = "typing"

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        if ir.path is None:
            return ir

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        def local_symbol_id(name: str) -> int | None:
            sym = ctx.get_exported_symbol(current_module, name)
            return sym.id.value if sym is not None else None

        def make_bool(value: bool, ref: PyIRNode) -> PyIRNode:
            return PyIRConstant(
                line=ref.line,
                offset=ref.offset,
                value=value,
            )

        compiler_directive_symids = collect_core_compiler_directive_local_symbol_ids(
            ir,
            ctx=ctx,
            current_module=current_module,
        )

        type_checking_local_ids: set[int] = set()
        typing_module_local_ids: set[int] = set()

        for n in ir.walk():
            if not isinstance(n, PyIRImport):
                continue

            modules = getattr(n, "modules", None) or []
            mod = modules[0] if isinstance(modules, list) and modules else ""
            names = getattr(n, "names", None)

            if (
                n.if_from
                and mod == FoldCompileTimeBoolConstsPass.TYPING_MODULE
                and names
            ):
                for orig, bound in iter_imported_names(names):
                    if orig == "TYPE_CHECKING":
                        sid = local_symbol_id(bound)
                        if sid is not None:
                            type_checking_local_ids.add(sid)

            if (not n.if_from) and mod == FoldCompileTimeBoolConstsPass.TYPING_MODULE:
                if names:
                    for _, bound in iter_imported_names(names):
                        sid = local_symbol_id(bound)
                        if sid is not None:
                            typing_module_local_ids.add(sid)

                else:
                    sid = local_symbol_id("typing")
                    if sid is not None:
                        typing_module_local_ids.add(sid)

        if (
            not compiler_directive_symids
            and not type_checking_local_ids
            and not typing_module_local_ids
        ):
            return ir

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return [rw(x, store=False) for x in body]

        def rw(node: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(node, PyIRSymLink):
                if store or node.is_store:
                    return node

                if node.symbol_id in type_checking_local_ids:
                    return make_bool(False, node)

                return node

            if isinstance(node, PyIRAttribute):
                node.value = rw(node.value, store=False)

                if store:
                    return node

                if isinstance(node.value, PyIRSymLink) and not node.value.is_store:
                    base_id = node.value.symbol_id

                    if node.attr == "DEBUG" and base_id in compiler_directive_symids:
                        return make_bool(bool(Py2GluaConfig.debug), node)

                    if (
                        node.attr == "TYPE_CHECKING"
                        and base_id in typing_module_local_ids
                    ):
                        return make_bool(False, node)

                return node

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

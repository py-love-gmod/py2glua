from __future__ import annotations

from typing import Iterable

from ....compiler.passes.analysis.symlinks import PyIRSymLink, SymLinkContext
from ....py.ir_dataclass import (
    LuaNil,
    PyIRAttribute,
    PyIRCall,
    PyIRConstant,
    PyIRFile,
    PyIRImport,
    PyIRNode,
)
from ..rewrite_utils import rewrite_expr_with_store_default, rewrite_stmt_block


class NilFoldPass:
    """
    Normalizes "nil" usage into a real IR constant.

    Goal:
      - replace reads of imported `nil` with `PyIRConstant(value=nil)`
      - replace `nil()` (no-arg call) with `PyIRConstant(value=nil)`
      - support `import py2glua.glua.core.types as t; t.nil` too

    This pass MUST run after symlink analysis/rewrite, because we rely on symbol_id.
    """

    TYPES_MODULE = "py2glua.glua.core.types"

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        if ir.path is None:
            return ir

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        def make_nil(ref: PyIRNode) -> PyIRConstant:
            return PyIRConstant(line=ref.line, offset=ref.offset, value=LuaNil)

        def iter_imported_names(
            names: list[str | tuple[str, str]],
        ) -> Iterable[tuple[str, str]]:
            for n in names:
                if isinstance(n, tuple):
                    yield n[0], n[1]
                else:
                    yield n, n

        def local_symbol_id(name: str) -> int | None:
            sym = ctx.get_exported_symbol(current_module, name)
            return sym.id.value if sym is not None else None

        nil_canon_sym = ctx.get_exported_symbol(NilFoldPass.TYPES_MODULE, "nil")
        nil_canon_id = nil_canon_sym.id.value if nil_canon_sym is not None else None

        nil_name_ids: set[int] = set()
        types_module_alias_ids: set[int] = set()

        for node in ir.walk():
            if not isinstance(node, PyIRImport):
                continue

            mod = ".".join(node.modules) if node.modules else ""
            if mod != NilFoldPass.TYPES_MODULE:
                continue

            if node.if_from:
                for orig, bound in iter_imported_names(node.names):
                    if orig != "nil":
                        continue

                    sid = local_symbol_id(bound)
                    if sid is not None:
                        nil_name_ids.add(sid)

                continue

            if node.names:
                for _orig, bound in iter_imported_names(node.names):
                    sid = local_symbol_id(bound)
                    if sid is not None:
                        types_module_alias_ids.add(sid)

        nil_ids: set[int] = set(nil_name_ids)
        if nil_canon_id is not None:
            nil_ids.add(nil_canon_id)

        if not nil_ids and not types_module_alias_ids:
            return ir

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(node, PyIRSymLink):
                if store or node.is_store:
                    return node

                if node.symbol_id in nil_ids:
                    return make_nil(node)

                return node

            if isinstance(node, PyIRCall):
                node.func = rw_expr(node.func, store=False)
                node.args_p = [rw_expr(a, store=False) for a in node.args_p]
                node.args_kw = {
                    k: rw_expr(v, store=False) for k, v in node.args_kw.items()
                }

                if (
                    not store
                    and not node.args_p
                    and not node.args_kw
                    and isinstance(node.func, PyIRSymLink)
                    and node.func.symbol_id in nil_ids
                ):
                    return make_nil(node)

                return node

            if isinstance(node, PyIRAttribute):
                node.value = rw_expr(node.value, store=False)

                if store:
                    return node

                if (
                    node.attr == "nil"
                    and isinstance(node.value, PyIRSymLink)
                    and not node.value.is_store
                    and node.value.symbol_id in types_module_alias_ids
                ):
                    return make_nil(node)

                return node

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

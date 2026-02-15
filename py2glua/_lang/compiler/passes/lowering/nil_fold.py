from __future__ import annotations

from typing import Iterable

from ....compiler.passes.analysis.symlinks import PyIRSymLink, SymLinkContext
from ....py.ir_dataclass import (
    LuaNil,
    PyIRAnnotatedAssign,
    PyIRAssign,
    PyIRAttribute,
    PyIRCall,
    PyIRConstant,
    PyIRFile,
    PyIRImport,
    PyIRNode,
    PyIRVarUse,
)
from ..common import (
    CORE_TYPES_MODULES,
    attr_chain_parts,
    collect_local_imported_symbol_ids,
    collect_symbol_ids_in_modules,
    is_expr_on_symbol_ids,
    resolve_symbol_ids_by_attr_chain,
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

    NIL_MODULES = CORE_TYPES_MODULES

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

        nil_name_ids: set[int] = collect_symbol_ids_in_modules(
            ctx,
            symbol_name="nil",
            modules=NilFoldPass.NIL_MODULES,
        )
        nil_name_ids |= collect_local_imported_symbol_ids(
            ir,
            ctx=ctx,
            current_module=current_module,
            imported_name="nil",
            allowed_modules=NilFoldPass.NIL_MODULES,
        )
        types_module_alias_ids: set[int] = set()

        for node in ir.walk():
            if not isinstance(node, PyIRImport):
                continue

            mod = ".".join(node.modules) if node.modules else ""
            if mod not in NilFoldPass.NIL_MODULES:
                continue

            if node.names:
                for _orig, bound in iter_imported_names(node.names):
                    sym = ctx.get_exported_symbol(current_module, bound)
                    sid = sym.id.value if sym is not None else None
                    if sid is not None:
                        types_module_alias_ids.add(sid)

        nil_ids: set[int] = set(nil_name_ids)

        if not nil_ids and not types_module_alias_ids:
            return ir

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            def is_nil_attr_chain(expr: PyIRNode) -> bool:
                parts = attr_chain_parts(expr)
                if not parts:
                    return False
                return tuple(parts[-4:]) == ("py2glua", "glua", "core", "nil") or tuple(
                    parts[-5:]
                ) == ("py2glua", "glua", "core", "types", "nil")

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
                    and (
                        (
                            isinstance(node.func, PyIRSymLink)
                            and node.func.symbol_id in nil_ids
                        )
                        or is_expr_on_symbol_ids(
                            node.func,
                            symbol_ids=nil_ids,
                            ctx=ctx,
                        )
                        or (
                            isinstance(node.func, PyIRAttribute)
                            and node.func.attr == "nil"
                            and (
                                any(
                                    sid in nil_ids
                                    for sid in resolve_symbol_ids_by_attr_chain(
                                        ctx,
                                        node.func,
                                    )
                                )
                                or is_nil_attr_chain(node.func)
                            )
                        )
                        or (
                            isinstance(node.func, PyIRVarUse)
                            and node.func.name == "nil"
                        )
                        or (
                            isinstance(node.func, PyIRSymLink)
                            and node.func.name == "nil"
                        )
                    )
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
                if node.attr == "nil" and is_nil_attr_chain(node):
                    return make_nil(node)

                return node

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)

        if current_module == "py2glua.glua.core.types":
            pruned: list[PyIRNode] = []
            for st in ir.body:
                if isinstance(st, PyIRAnnotatedAssign) and st.name == "nil":
                    continue
                if isinstance(st, PyIRAssign) and len(st.targets) == 1:
                    t = st.targets[0]
                    tname: str | None = None
                    if isinstance(t, PyIRVarUse):
                        tname = t.name
                    elif isinstance(t, PyIRSymLink):
                        tname = t.name
                    if tname == "nil":
                        continue
                pruned.append(st)
            ir.body = pruned

        return ir

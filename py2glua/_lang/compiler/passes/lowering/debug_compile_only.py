from __future__ import annotations

from typing import Final

from .....config import Py2GluaConfig
from ....py.ir_dataclass import (
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
    PyIRVarCreate,
)
from ..rewrite_utils import (
    rewrite_expr_with_store_default,
    rewrite_stmt_block,
)
from ..analysis.symlinks import PyIRSymLink, SymLinkContext
from ..common import (
    canon_path,
    collect_decl_symid_cache,
    collect_core_compiler_directive_local_symbol_ids,
    is_attr_on_symbol_ids,
)


class CollectDebugCompileOnlyDeclsPass:
    """
    Lowering-pass (phase 1):
    Collects symids of defs decorated with @CompilerDirective.debug_compile_only()
    (ONLY if decorator base is proven to be core CompilerDirective binding).

    Writes into ctx:
      - ctx._debug_compile_only_symids: set[int]
      - ctx._debug_compile_only_symid_by_decl_cache: dict[(Path, line, name) -> symid]
    """

    _DECL_CACHE_FIELD: Final[str] = "_debug_compile_only_symid_by_decl_cache"
    _DEBUG_ONLY_NAME: Final[str] = "debug_compile_only"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectDebugCompileOnlyDeclsPass.run ожидает PyIRFile")

        if bool(Py2GluaConfig.debug):
            return ir

        s = ctx._debug_compile_only_symids

        if ir.path is None:
            return ir

        ctx.ensure_module_index()
        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        symid_by_decl = collect_decl_symid_cache(ctx, cache_field=cls._DECL_CACHE_FIELD)
        fp = canon_path(ir.path)
        if fp is None:
            return ir

        cd_local_symids = collect_core_compiler_directive_local_symbol_ids(
            ir=ir,
            ctx=ctx,
            current_module=current_module,
        )
        for n in ir.walk():
            if isinstance(n, (PyIRFunctionDef, PyIRClassDef)):
                if n.line is None:
                    continue

                if not any(
                    is_attr_on_symbol_ids(
                        d.exper,
                        attr=cls._DEBUG_ONLY_NAME,
                        symbol_ids=cd_local_symids,
                        ctx=ctx,
                    )
                    for d in (n.decorators or [])
                ):
                    continue

                sid = symid_by_decl.get((fp, n.line, n.name))
                if sid is not None:
                    s.add(int(sid))

        return ir


class RewriteAndStripDebugCompileOnlyPass:
    """
    Lowering-pass (phase 2, release only):

    - Removes defs whose symid is in ctx._debug_compile_only_symids.
    - Rewrites read-uses of debug-only symbols to nil (PyIRConstant(value=None)).
    - Rewrites calls to debug-only symbols:
        * if call is a standalone statement => removes it from block
        * otherwise => replaces call expr with nil

    NOTE:
      This pass does NOT try to preserve side effects of debug-only call args.
      It's "debug only": in release it is assumed safe to erase.
    """

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("RewriteAndStripDebugCompileOnlyPass.run ожидает PyIRFile")

        if bool(Py2GluaConfig.debug):
            return ir

        debug_symids = ctx._debug_compile_only_symids
        if not debug_symids:
            return ir

        if ir.path is None:
            return ir

        symid_by_decl = collect_decl_symid_cache(
            ctx,
            cache_field=CollectDebugCompileOnlyDeclsPass._DECL_CACHE_FIELD,
        )
        fp = canon_path(ir.path)
        if fp is None:
            return ir

        def node_symid(n: PyIRNode) -> int | None:
            if not isinstance(n, (PyIRFunctionDef, PyIRClassDef)):
                return None
            if n.line is None:
                return None
            return symid_by_decl.get((fp, n.line, n.name))

        def make_nil(ref: PyIRNode) -> PyIRNode:
            return PyIRConstant(line=ref.line, offset=ref.offset, value=None)

        def is_debug_def_node(st: PyIRNode) -> bool:
            sid = node_symid(st)
            return sid is not None and int(sid) in debug_symids

        def is_debug_symlink_read(n: PyIRSymLink, store: bool) -> bool:
            if store or n.is_store:
                return False
            return int(n.symbol_id) in debug_symids

        def is_debug_call_expr(expr: PyIRNode) -> bool:
            if not isinstance(expr, PyIRCall):
                return False
            f = expr.func
            if isinstance(f, PyIRSymLink):
                return int(f.symbol_id) in debug_symids
            return False

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            removed_names: set[str] = set()
            removed_nodes: set[int] = set()

            for st in body:
                if not is_debug_def_node(st):
                    continue
                removed_nodes.add(id(st))
                if isinstance(st, (PyIRFunctionDef, PyIRClassDef)):
                    removed_names.add(st.name)

            def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
                if id(st) in removed_nodes:
                    return []

                if isinstance(st, PyIRVarCreate) and st.name in removed_names:
                    return []

                if isinstance(st, PyIRCall) and is_debug_call_expr(st):
                    return []

                return [rw_expr(st, False)]

            return rewrite_stmt_block(body, rw_stmt)

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(node, PyIRSymLink):
                if is_debug_symlink_read(node, store):
                    return make_nil(node)
                return node

            if isinstance(node, PyIRCall) and is_debug_call_expr(node):
                return make_nil(node)

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

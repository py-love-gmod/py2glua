from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final, Iterator, Sequence

from .....config import Py2GluaConfig
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRDecorator,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRNode,
    PyIRVarCreate,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ...compiler_ir import PyIRFunctionExpr
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


class CollectDebugCompileOnlyDeclsPass:
    """
    Lowering-pass (phase 1):
    Collects symids of defs decorated with @CompilerDirective.debug_compile_only()
    (ONLY if decorator base is proven to be CompilerDirective binding in this file).

    Writes into ctx:
      - ctx._debug_compile_only_symids: set[int]
      - ctx._debug_compile_only_symid_by_decl_cache: dict[(Path, line, name) -> symid]
    """

    _SET_FIELD: Final[str] = "_debug_compile_only_symids"
    _DECL_CACHE_FIELD: Final[str] = "_debug_compile_only_symid_by_decl_cache"

    _CD_NAME: Final[str] = "CompilerDirective"
    _DEBUG_ONLY_NAME: Final[str] = "debug_compile_only"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectDebugCompileOnlyDeclsPass.run expects PyIRFile")

        if bool(Py2GluaConfig.debug):
            return ir

        s = getattr(ctx, cls._SET_FIELD, None)
        if not isinstance(s, set):
            s = set()
            setattr(ctx, cls._SET_FIELD, s)

        if ir.path is None:
            return ir

        ctx.ensure_module_index()
        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        symid_by_decl = cls._get_symid_by_decl(ctx)
        fp = cls._canon_path(ir.path)
        if fp is None:
            return ir

        cd_local_symids = cls._collect_local_compiler_directive_symids(
            ir=ir,
            ctx=ctx,
            current_module=current_module,
            file_path=fp,
            symid_by_decl=symid_by_decl,
        )
        if not cd_local_symids:
            return ir

        for n in ir.walk():
            if isinstance(n, (PyIRFunctionDef, PyIRClassDef)):
                if n.line is None:
                    continue

                if not cls._has_debug_only(n.decorators, cd_local_symids):
                    continue

                sid = symid_by_decl.get((fp, n.line, n.name))
                if sid is not None:
                    s.add(int(sid))

        return ir

    @staticmethod
    def _canon_path(p: Path | None) -> Path | None:
        if p is None:
            return None
        try:
            return p.resolve()
        except Exception:
            return p

    @classmethod
    def _get_symid_by_decl(
        cls, ctx: SymLinkContext
    ) -> dict[tuple[Path, int, str], int]:
        cached = getattr(ctx, cls._DECL_CACHE_FIELD, None)
        if isinstance(cached, dict):
            return cached

        out: dict[tuple[Path, int, str], int] = {}
        for sym in ctx.symbols.values():
            if sym.decl_file is None or sym.decl_line is None:
                continue
            p = cls._canon_path(sym.decl_file)
            if p is None:
                continue
            out[(p, sym.decl_line, sym.name)] = sym.id.value

        setattr(ctx, cls._DECL_CACHE_FIELD, out)
        return out

    @staticmethod
    def _iter_imported_names(
        names: Sequence[str | tuple[str, str]] | None,
    ) -> Iterator[tuple[str, str]]:
        if not names:
            return
        for n in names:
            if isinstance(n, tuple):
                yield n[0], n[1]
            else:
                yield n, n

    @classmethod
    def _collect_local_compiler_directive_symids(
        cls,
        *,
        ir: PyIRFile,
        ctx: SymLinkContext,
        current_module: str,
        file_path: Path,
        symid_by_decl: dict[tuple[Path, int, str], int],
    ) -> set[int]:
        """
        Returns symids of names in this file that refer to CompilerDirective class:
          - local class def `class CompilerDirective:`
          - from-import `from X import CompilerDirective as CD`
          - from-import `from X import CompilerDirective`
        """
        out: set[int] = set()

        def local_symbol_id(name: str) -> int | None:
            sym = ctx.get_exported_symbol(current_module, name)
            return sym.id.value if sym is not None else None

        for n in ir.walk():
            if (
                isinstance(n, PyIRClassDef)
                and n.name == cls._CD_NAME
                and n.line is not None
            ):
                sid = symid_by_decl.get((file_path, n.line, n.name))
                if sid is not None:
                    out.add(int(sid))
                break

        for n in ir.walk():
            if not isinstance(n, PyIRImport):
                continue

            modules = getattr(n, "modules", None) or []
            mod = modules[0] if isinstance(modules, list) and modules else ""
            names = getattr(n, "names", None)

            if not n.if_from or not mod or not names:
                continue

            for orig, bound in cls._iter_imported_names(names):
                if orig != cls._CD_NAME:
                    continue
                sid = local_symbol_id(bound)
                if sid is not None:
                    out.add(int(sid))

        return out

    @classmethod
    def _has_debug_only(
        cls,
        decorators: list[PyIRDecorator],
        cd_local_symids: set[int],
    ) -> bool:
        for d in decorators or []:
            if cls._is_debug_only_expr(d.exper, cd_local_symids):
                return True
        return False

    @classmethod
    def _is_debug_only_expr(
        cls,
        expr: PyIRNode,
        cd_local_symids: set[int],
    ) -> bool:

        while isinstance(expr, PyIRCall):
            expr = expr.func

        if not isinstance(expr, PyIRAttribute):
            return False
        if expr.attr != cls._DEBUG_ONLY_NAME:
            return False

        base = expr.value
        if isinstance(base, PyIRSymLink):
            return int(base.symbol_id) in cd_local_symids

        return False


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

    _SET_FIELD: Final[str] = "_debug_compile_only_symids"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("RewriteAndStripDebugCompileOnlyPass.run expects PyIRFile")

        if bool(Py2GluaConfig.debug):
            return ir

        debug_symids = getattr(ctx, cls._SET_FIELD, None)
        if not isinstance(debug_symids, set) or not debug_symids:
            return ir

        if ir.path is None:
            return ir

        symid_by_decl = CollectDebugCompileOnlyDeclsPass._get_symid_by_decl(ctx)
        fp = CollectDebugCompileOnlyDeclsPass._canon_path(ir.path)
        if fp is None:
            return ir

        def node_symid(n: PyIRNode) -> int | None:
            if isinstance(n, (PyIRFunctionDef, PyIRClassDef)):
                if n.line is None:
                    return None
                return symid_by_decl.get((fp, n.line, n.name))
            return None

        def make_nil(ref: PyIRNode) -> PyIRNode:

            return PyIRConstant(line=ref.line, offset=ref.offset, value=None)

        def is_debug_def_node(st: PyIRNode) -> bool:
            sid = node_symid(st)
            return (sid is not None) and (int(sid) in debug_symids)

        def is_debug_symlink_read(n: PyIRSymLink, *, store: bool) -> bool:
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

        def rw_expr(node: PyIRNode, *, store: bool) -> PyIRNode:
            if isinstance(node, PyIRSymLink):
                if is_debug_symlink_read(node, store=store):
                    return make_nil(node)
                return node

            if isinstance(node, PyIRAttribute):
                node.value = rw_expr(node.value, store=False)
                return node

            if isinstance(node, PyIRCall):
                if is_debug_call_expr(node):
                    return make_nil(node)
                node.func = rw_expr(node.func, store=False)
                node.args_p = [rw_expr(a, store=False) for a in node.args_p]
                node.args_kw = {
                    k: rw_expr(v, store=False) for k, v in node.args_kw.items()
                }
                return node

            if isinstance(node, PyIRAssign):
                node.targets = [rw_expr(t, store=True) for t in node.targets]
                node.value = rw_expr(node.value, store=False)
                return node

            if isinstance(node, PyIRAugAssign):
                node.target = rw_expr(node.target, store=True)
                node.value = rw_expr(node.value, store=False)
                return node

            if isinstance(node, PyIRFor):
                node.target = rw_expr(node.target, store=True)
                node.iter = rw_expr(node.iter, store=False)
                node.body = rw_block(node.body)
                return node

            if isinstance(node, PyIRWithItem):
                node.context_expr = rw_expr(node.context_expr, store=False)
                if node.optional_vars is not None:
                    node.optional_vars = rw_expr(node.optional_vars, store=True)
                return node

            if isinstance(node, PyIRWith):
                node.items = [rw_expr(it, store=False) for it in node.items]  # type: ignore
                node.body = rw_block(node.body)
                return node

            if isinstance(node, PyIRIf):
                node.test = rw_expr(node.test, store=False)
                node.body = rw_block(node.body)
                node.orelse = rw_block(node.orelse)
                return node

            if isinstance(node, PyIRWhile):
                node.test = rw_expr(node.test, store=False)
                node.body = rw_block(node.body)
                return node

            if isinstance(node, PyIRFunctionDef):
                node.decorators = [rw_expr(d, store=False) for d in node.decorators]  # type: ignore
                node.body = rw_block(node.body)
                return node

            if isinstance(node, PyIRFunctionExpr):
                node.body = rw_block(node.body)
                return node

            if isinstance(node, PyIRClassDef):
                node.decorators = [rw_expr(d, store=False) for d in node.decorators]  # type: ignore
                node.bases = [rw_expr(b, store=False) for b in node.bases]
                node.body = rw_block(node.body)
                return node

            if is_dataclass(node):
                for f in fields(node):
                    v = getattr(node, f.name)
                    if isinstance(v, PyIRNode):
                        setattr(node, f.name, rw_expr(v, store=False))
                    elif isinstance(v, list):
                        setattr(
                            node,
                            f.name,
                            [
                                rw_expr(x, store=False)
                                if isinstance(x, PyIRNode)
                                else x
                                for x in v
                            ],
                        )
                    elif isinstance(v, dict):
                        setattr(
                            node,
                            f.name,
                            {
                                k: (
                                    rw_expr(x, store=False)
                                    if isinstance(x, PyIRNode)
                                    else x
                                )
                                for k, x in v.items()
                            },
                        )
                return node

            return node

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            removed_names: set[str] = set()
            removed_nodes: set[int] = set()

            for st in body:
                if is_debug_def_node(st):
                    removed_nodes.add(id(st))
                    if isinstance(st, (PyIRFunctionDef, PyIRClassDef)):
                        removed_names.add(st.name)

            out: list[PyIRNode] = []
            for st in body:
                if id(st) in removed_nodes:
                    continue

                if isinstance(st, PyIRVarCreate) and st.name in removed_names:
                    continue

                if isinstance(st, PyIRCall) and is_debug_call_expr(st):
                    continue

                out.append(rw_expr(st, store=False))

            return out

        ir.body = rw_block(ir.body)
        return ir

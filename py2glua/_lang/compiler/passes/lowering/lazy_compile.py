from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final

from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRDecorator,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRVarCreate,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


class CountSymlinkUsesPass:
    """
    Lowering-pass: counts "read-uses" of symlinks across ALL files
    (because compiler runs this pass for every file before the next pass).

    Writes into ctx:
      - ctx.symlink_use_count: dict[int, int]  (symbol_id -> read count)

    Notes:
      - Counts only reads: PyIRSymLink with is_store == False.
      - Does NOT count self-uses inside the same def/class nest
        (so recursion / internal references don't keep lazy code alive).
      - Additionally counts method uses for:
            <ClassSymLink>.<method_name>
        by resolving <method_name> in the class body scope (when known).
        If class body scope isn't known yet, it postpones and resolves later
        when that class is encountered during the same pass (another file).
    """

    _COUNT_FIELD: Final[str] = "symlink_use_count"

    _DECL_CACHE_FIELD: Final[str] = "_symlink_symid_by_decl_cache"

    _CLS_SCOPE_FIELD: Final[str] = "_symlink_class_scope_by_symid"
    _PENDING_ATTR_FIELD: Final[str] = "_symlink_pending_attr_uses"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CountSymlinkUsesLoweringPass.run expects PyIRFile")

        counts = getattr(ctx, cls._COUNT_FIELD, None)
        if not isinstance(counts, dict):
            counts = {}
            setattr(ctx, cls._COUNT_FIELD, counts)

        symid_by_decl = cls._get_symid_by_decl(ctx)

        cls_scope_by_symid = getattr(ctx, cls._CLS_SCOPE_FIELD, None)
        if not isinstance(cls_scope_by_symid, dict):
            cls_scope_by_symid = {}
            setattr(ctx, cls._CLS_SCOPE_FIELD, cls_scope_by_symid)

        pending_attr = getattr(ctx, cls._PENDING_ATTR_FIELD, None)
        if not isinstance(pending_attr, dict):
            pending_attr = {}
            setattr(ctx, cls._PENDING_ATTR_FIELD, pending_attr)

        # 1) Register classes from this file -> class_symid -> class_body_scope
        cls._register_classes_from_file(
            ir=ir,
            ctx=ctx,
            symid_by_decl=symid_by_decl,
            cls_scope_by_symid=cls_scope_by_symid,
            pending_attr=pending_attr,
            counts=counts,
        )

        # 2) Count uses inside this file (self-uses excluded via def_stack)
        cls._count_in_file(
            ir=ir,
            ctx=ctx,
            symid_by_decl=symid_by_decl,
            cls_scope_by_symid=cls_scope_by_symid,
            pending_attr=pending_attr,
            counts=counts,
        )

        return ir

    # -------------------------
    # helpers: canonical paths
    # -------------------------

    @staticmethod
    def _canon_path(p: Path | None) -> Path | None:
        if p is None:
            return None
        try:
            return p.resolve()
        except Exception:
            return p

    # -------------------------
    # helpers: decl -> symid
    # -------------------------

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

    # -------------------------
    # helpers: class registry + pending resolution
    # -------------------------

    @classmethod
    def _register_classes_from_file(
        cls,
        *,
        ir: PyIRFile,
        ctx: SymLinkContext,
        symid_by_decl: dict[tuple[Path, int, str], int],
        cls_scope_by_symid: dict[int, object],
        pending_attr: dict[tuple[int, str], int],
        counts: dict[int, int],
    ) -> None:
        if ir.path is None:
            return
        fp = cls._canon_path(ir.path)
        if fp is None:
            return

        # Find class defs in this file, map to class_symid, then map to class body scope
        for n in ir.walk():
            if not isinstance(n, PyIRClassDef):
                continue
            if n.line is None:
                continue

            class_symid = symid_by_decl.get((fp, n.line, n.name))
            if class_symid is None:
                continue
            if class_symid in cls_scope_by_symid:
                continue

            uid = ctx.uid(n)
            cls_scope = ctx.cls_body_scope.get(uid)
            if cls_scope is None:
                continue

            cls_scope_by_symid[class_symid] = cls_scope

            # Flush pending attribute uses for this class now that we know the scope
            cls._flush_pending_for_class(
                ctx=ctx,
                class_symid=class_symid,
                cls_scope=cls_scope,
                pending_attr=pending_attr,
                counts=counts,
            )

    @classmethod
    def _flush_pending_for_class(
        cls,
        *,
        ctx: SymLinkContext,
        class_symid: int,
        cls_scope: object,
        pending_attr: dict[tuple[int, str], int],
        counts: dict[int, int],
    ) -> None:
        # Copy keys because we mutate
        to_flush = [k for k in pending_attr.keys() if k[0] == class_symid]
        for key in to_flush:
            _, attr = key
            ncount = int(pending_attr.get(key, 0))
            if ncount <= 0:
                pending_attr.pop(key, None)
                continue

            link = ctx.resolve(scope=cls_scope, name=attr)  # type: ignore[arg-type]
            if link is not None:
                sid = int(link.target.value)
                counts[sid] = counts.get(sid, 0) + ncount

            pending_attr.pop(key, None)

    # -------------------------
    # counting
    # -------------------------

    @classmethod
    def _count_in_file(
        cls,
        *,
        ir: PyIRFile,
        ctx: SymLinkContext,
        symid_by_decl: dict[tuple[Path, int, str], int],
        cls_scope_by_symid: dict[int, object],
        pending_attr: dict[tuple[int, str], int],
        counts: dict[int, int],
    ) -> None:
        fp = cls._canon_path(ir.path) if ir.path is not None else None

        def_stack: list[int] = []

        def inc(symid: int, n: int = 1) -> None:
            counts[symid] = counts.get(symid, 0) + n

        def node_symid(n: PyIRNode) -> int | None:
            if fp is None:
                return None
            if isinstance(n, (PyIRFunctionDef, PyIRClassDef)):
                if n.line is None:
                    return None
                return symid_by_decl.get((fp, n.line, n.name))
            return None

        def count_class_attr_use(class_symid: int, attr: str) -> None:
            cls_scope = cls_scope_by_symid.get(class_symid)
            if cls_scope is not None:
                link = ctx.resolve(scope=cls_scope, name=attr)  # type: ignore[arg-type]
                if link is not None:
                    inc(int(link.target.value))
                return

            # postpone until class is registered from its file
            key = (class_symid, attr)
            pending_attr[key] = pending_attr.get(key, 0) + 1

        def walk(node: PyIRNode, *, store: bool) -> None:
            # --- symlink read count
            if isinstance(node, PyIRSymLink):
                if not node.is_store:
                    if node.symbol_id not in def_stack:
                        inc(int(node.symbol_id))
                return

            # --- attribute: count base; if it's a class symlink and it's read, count method symbol too
            if isinstance(node, PyIRAttribute):
                walk(node.value, store=False)

                if not store and isinstance(node.value, PyIRSymLink):
                    # class symbol id might represent an actual class (we'll resolve method in class scope)
                    count_class_attr_use(int(node.value.symbol_id), node.attr)

                return

            # --- call
            if isinstance(node, PyIRCall):
                walk(node.func, store=False)
                for a in node.args_p:
                    walk(a, store=False)
                for a in node.args_kw.values():
                    walk(a, store=False)
                return

            # --- assignment / stores
            if isinstance(node, PyIRAssign):
                for t in node.targets:
                    walk(t, store=True)
                walk(node.value, store=False)
                return

            if isinstance(node, PyIRAugAssign):
                walk(node.target, store=True)
                walk(node.value, store=False)
                return

            if isinstance(node, PyIRFor):
                walk(node.target, store=True)
                walk(node.iter, store=False)
                for b in node.body:
                    walk(b, store=False)
                return

            if isinstance(node, PyIRWithItem):
                walk(node.context_expr, store=False)
                if node.optional_vars is not None:
                    walk(node.optional_vars, store=True)
                return

            if isinstance(node, PyIRWith):
                for it in node.items:
                    walk(it, store=False)
                for b in node.body:
                    walk(b, store=False)
                return

            # --- control flow blocks
            if isinstance(node, PyIRIf):
                walk(node.test, store=False)
                for b in node.body:
                    walk(b, store=False)
                for b in node.orelse:
                    walk(b, store=False)
                return

            if isinstance(node, PyIRWhile):
                walk(node.test, store=False)
                for b in node.body:
                    walk(b, store=False)
                return

            # --- defs: exclude self-uses inside their own body
            if isinstance(node, PyIRFunctionDef):
                sid = node_symid(node)
                if sid is not None:
                    def_stack.append(sid)

                for d in node.decorators:
                    walk(d, store=False)
                for b in node.body:
                    walk(b, store=False)

                if sid is not None:
                    def_stack.pop()

                return

            if isinstance(node, PyIRClassDef):
                sid = node_symid(node)
                if sid is not None:
                    def_stack.append(sid)

                for d in node.decorators:
                    walk(d, store=False)
                for b in node.bases:
                    walk(b, store=False)
                for x in node.body:
                    walk(x, store=False)

                if sid is not None:
                    def_stack.pop()

                return

            if isinstance(node, PyIRDecorator):
                walk(node.exper, store=False)
                for a in node.args_p:
                    walk(a, store=False)
                for a in node.args_kw.values():
                    walk(a, store=False)
                return

            # PyIRVarCreate itself is not a "use"
            if isinstance(node, PyIRVarCreate):
                return

            # --- generic dataclass descent
            if not is_dataclass(node):
                return

            for f in fields(node):
                v = getattr(node, f.name)
                if isinstance(v, PyIRNode):
                    walk(v, store=False)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, PyIRNode):
                            walk(x, store=False)
                elif isinstance(v, dict):
                    for x in v.values():
                        if isinstance(x, PyIRNode):
                            walk(x, store=False)

        for st in ir.body:
            walk(st, store=False)


class StripLazyCompileUnusedDefsPass:
    """
    Lowering-pass: removes defs decorated with @lazy_compile if use_count == 0.

    Requires:
      - CountSymlinkUsesLoweringPass must be placed BEFORE this pass in lowering_passes,
        so ctx.symlink_use_count already reflects the whole project.
    """

    _COUNT_FIELD: Final[str] = "symlink_use_count"
    _LAZY_NAME: Final[str] = "lazy_compile"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError(
                "StripLazyCompileUnusedDefsLoweringPass.run expects PyIRFile"
            )

        if ir.path is None:
            return ir

        counts = getattr(ctx, cls._COUNT_FIELD, {})
        if not isinstance(counts, dict):
            counts = {}

        symid_by_decl = CountSymlinkUsesPass._get_symid_by_decl(ctx)

        file_path = CountSymlinkUsesPass._canon_path(ir.path)
        if file_path is None:
            return ir

        def node_symid(n: PyIRNode) -> int | None:
            if isinstance(n, (PyIRFunctionDef, PyIRClassDef)):
                if n.line is None:
                    return None
                return symid_by_decl.get((file_path, n.line, n.name))
            return None

        def use_count(symid: int | None) -> int:
            if symid is None:
                return 0
            return int(counts.get(symid, 0))

        def strip_block(body: list[PyIRNode]) -> list[PyIRNode]:
            removed_names: set[str] = set()
            removed_nodes: set[int] = set()

            # first pass: decide removals in this block
            for st in body:
                if isinstance(
                    st, (PyIRFunctionDef, PyIRClassDef)
                ) and cls._has_lazy_compile(st.decorators):
                    sid = node_symid(st)
                    if sid is not None and use_count(sid) == 0:
                        removed_nodes.add(id(st))
                        removed_names.add(st.name)

            # second pass: rebuild, recurse
            out: list[PyIRNode] = []
            for st in body:
                if id(st) in removed_nodes:
                    continue

                # also remove matching declarations (local name) if present in same block
                if isinstance(st, PyIRVarCreate) and st.name in removed_names:
                    continue

                if isinstance(st, PyIRFunctionDef):
                    st.body = strip_block(st.body)
                    out.append(st)
                    continue

                if isinstance(st, PyIRClassDef):
                    st.body = strip_block(st.body)
                    out.append(st)
                    continue

                out.append(cls._rw_generic(st, strip_block))

            return out

        ir.body = strip_block(ir.body)
        return ir

    @classmethod
    def _rw_generic(cls, node: PyIRNode, strip_block_fn) -> PyIRNode:
        """
        Conservative recursion: rewrites nested list[PyIRNode] fields,
        plus generic PyIRNode fields, without assuming emitter semantics.
        """
        if isinstance(node, (PyIRFunctionDef, PyIRClassDef)):
            return node

        if not is_dataclass(node):
            return node

        for f in fields(node):
            v = getattr(node, f.name)

            if isinstance(v, list) and v and all(isinstance(x, PyIRNode) for x in v):
                setattr(node, f.name, strip_block_fn(v))
                continue

            if isinstance(v, PyIRNode):
                setattr(node, f.name, cls._rw_generic(v, strip_block_fn))
                continue

            if isinstance(v, list):
                out_list = []
                changed = False
                for x in v:
                    if isinstance(x, PyIRNode):
                        nx = cls._rw_generic(x, strip_block_fn)
                        changed = changed or (nx is not x)
                        out_list.append(nx)
                    else:
                        out_list.append(x)
                if changed:
                    setattr(node, f.name, out_list)
                continue

            if isinstance(v, dict):
                out_dict = {}
                changed = False
                for k, x in v.items():
                    nk = (
                        cls._rw_generic(k, strip_block_fn)
                        if isinstance(k, PyIRNode)
                        else k
                    )
                    nx = (
                        cls._rw_generic(x, strip_block_fn)
                        if isinstance(x, PyIRNode)
                        else x
                    )
                    changed = changed or (nk is not k) or (nx is not x)
                    out_dict[nk] = nx
                if changed:
                    setattr(node, f.name, out_dict)

        return node

    @classmethod
    def _has_lazy_compile(cls, decorators: list[PyIRDecorator]) -> bool:
        for d in decorators or []:
            if cls._is_lazy_expr(d.exper):
                return True
        return False

    @classmethod
    def _is_lazy_expr(cls, expr: PyIRNode) -> bool:
        if isinstance(expr, PyIRCall):
            return cls._is_lazy_expr(expr.func)

        if isinstance(expr, PyIRAttribute):
            return expr.attr == cls._LAZY_NAME

        if isinstance(expr, PyIRSymLink):
            return expr.name == cls._LAZY_NAME

        # avoid importing PyIRVarUse here just for isinstance
        if type(expr).__name__ == "PyIRVarUse":
            return getattr(expr, "name", None) == cls._LAZY_NAME

        return False

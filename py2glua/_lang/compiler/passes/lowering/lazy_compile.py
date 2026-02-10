from __future__ import annotations

from dataclasses import fields, is_dataclass
from dataclasses import fields as dc_fields
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
from ..analysis.symlinks import (
    PyIRSymLink,
    ScopeId,
    SymLinkContext,
    get_cached_symlink_usage_stats,
)


class CountSymlinkUsesPass:
    _COUNT_FIELD: Final[str] = "symlink_use_count"
    _DECL_CACHE_FIELD: Final[str] = "_symlink_symid_by_decl_cache"

    _CLS_SCOPE_FIELD: Final[str] = "_symlink_class_scope_by_symid"
    _PENDING_ATTR_FIELD: Final[str] = "_symlink_pending_attr_uses_grouped"
    _ATTR_CACHE_FIELD: Final[str] = "_symlink_attr_resolve_cache"

    _FIELDS_CACHE: dict[type, tuple] = {}

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CountSymlinkUsesPass.run expects PyIRFile")

        counts = getattr(ctx, cls._COUNT_FIELD, None)
        if not isinstance(counts, dict):
            counts = {}
            setattr(ctx, cls._COUNT_FIELD, counts)

        symid_by_decl = cls._get_symid_by_decl(ctx)

        cls_scope_by_symid = getattr(ctx, cls._CLS_SCOPE_FIELD, None)
        if not isinstance(cls_scope_by_symid, dict):
            cls_scope_by_symid = {}
            setattr(ctx, cls._CLS_SCOPE_FIELD, cls_scope_by_symid)

        # grouped: class_symid -> { attr -> count }
        pending_attr = getattr(ctx, cls._PENDING_ATTR_FIELD, None)
        if not isinstance(pending_attr, dict):
            pending_attr = {}
            setattr(ctx, cls._PENDING_ATTR_FIELD, pending_attr)

        # cache: (class_symid, attr) -> method_symid | None
        attr_cache = getattr(ctx, cls._ATTR_CACHE_FIELD, None)
        if not isinstance(attr_cache, dict):
            attr_cache = {}
            setattr(ctx, cls._ATTR_CACHE_FIELD, attr_cache)

        cls._count_and_register_in_file(
            ir=ir,
            ctx=ctx,
            symid_by_decl=symid_by_decl,
            cls_scope_by_symid=cls_scope_by_symid,
            pending_attr=pending_attr,
            attr_cache=attr_cache,
            counts=counts,
        )

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

    @classmethod
    def _dc_fields_cached(cls, node: object) -> tuple:
        t = type(node)
        cached = cls._FIELDS_CACHE.get(t)
        if cached is None:
            if not is_dataclass(t):
                cached = ()
            else:
                cached = dc_fields(t)
            cls._FIELDS_CACHE[t] = cached

        return cached

    @classmethod
    def _count_and_register_in_file(
        cls,
        *,
        ir: PyIRFile,
        ctx: SymLinkContext,
        symid_by_decl: dict[tuple[Path, int, str], int],
        cls_scope_by_symid: dict[int, ScopeId],
        pending_attr: dict[int, dict[str, int]],
        attr_cache: dict[tuple[int, str], int | None],
        counts: dict[int, int],
    ) -> None:
        fp = cls._canon_path(ir.path) if ir.path is not None else None

        usage = get_cached_symlink_usage_stats(ir, ctx, skip_def_reads=True)

        def inc(symid: int, n: int = 1) -> None:
            counts[symid] = counts.get(symid, 0) + n

        for sid, ncount in usage.reads.items():
            if ncount > 0:
                inc(int(sid), int(ncount))

        def resolve_class_attr_use(class_symid: int, attr: str, n: int = 1) -> None:
            key = (class_symid, attr)

            if key in attr_cache:
                sid = attr_cache[key]
                if sid is not None:
                    inc(int(sid), n)

                return

            cls_scope = cls_scope_by_symid.get(class_symid)
            if cls_scope is None:
                bucket = pending_attr.get(class_symid)
                if bucket is None:
                    bucket = {}
                    pending_attr[class_symid] = bucket

                bucket[attr] = bucket.get(attr, 0) + n
                return

            link = ctx.resolve(scope=cls_scope, name=attr)
            if link is None:
                attr_cache[key] = None
                return

            sid = int(link.target.value)
            attr_cache[key] = sid
            inc(sid, n)

        def register_class_if_needed(node: PyIRClassDef) -> None:
            if fp is None or node.line is None:
                return

            class_symid = symid_by_decl.get((fp, node.line, node.name))
            if class_symid is None:
                return

            if class_symid in cls_scope_by_symid:
                return

            uid = ctx.uid(node)
            cls_scope = ctx.cls_body_scope.get(uid)
            if cls_scope is None:
                return

            cls_scope_by_symid[class_symid] = cls_scope

            bucket = pending_attr.pop(class_symid, None)
            if not bucket:
                return

            for attr, ncount in bucket.items():
                if ncount <= 0:
                    continue

                resolve_class_attr_use(class_symid, attr, n=ncount)

        def walk(node: PyIRNode, *, store: bool) -> None:
            if isinstance(node, PyIRClassDef):
                register_class_if_needed(node)

            if isinstance(node, PyIRSymLink):
                return

            if isinstance(node, PyIRAttribute):
                walk(node.value, store=False)

                if not store and isinstance(node.value, PyIRSymLink):
                    resolve_class_attr_use(int(node.value.symbol_id), node.attr)

                return

            if isinstance(node, PyIRCall):
                walk(node.func, store=False)
                for a in node.args_p:
                    walk(a, store=False)

                for a in node.args_kw.values():
                    walk(a, store=False)

                return

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

            if isinstance(node, PyIRFunctionDef):
                for d in node.decorators:
                    walk(d, store=False)

                for b in node.body:
                    walk(b, store=False)

                return

            if isinstance(node, PyIRClassDef):
                for d in node.decorators:
                    walk(d, store=False)

                for b in node.bases:
                    walk(b, store=False)

                for x in node.body:
                    walk(x, store=False)

                return

            if isinstance(node, PyIRDecorator):
                walk(node.exper, store=False)
                for a in node.args_p:
                    walk(a, store=False)

                for a in node.args_kw.values():
                    walk(a, store=False)

                return

            if isinstance(node, PyIRVarCreate):
                return

            if not hasattr(node, "__dataclass_fields__"):
                return

            for f in cls._dc_fields_cached(node):
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

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final

from .....config import Py2GluaConfig
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRComment,
    PyIRDecorator,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRNode,
    PyIRVarCreate,
    PyIRVarStorage,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ...compiler_ir import PyIRDo, PyIRFunctionExpr
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


class ResolveSymlinksToNamespaceProjectPass:
    """
    Project-pass: finalizes name resolution for codegen WITHOUT env shims.
    """

    _MAX_RAW_LINE: Final[int] = 160
    _SKIP_AUTORUN_TOPDIRS: Final[tuple[str, ...]] = ("entities", "weapons")

    @staticmethod
    def _is_simple_lua_name(name: str) -> bool:
        if not name or "." in name or ":" in name:
            return False

        c0 = name[0]
        if not (c0.isalpha() or c0 == "_"):
            return False

        for ch in name[1:]:
            if not (ch.isalnum() or ch == "_"):
                return False

        return True

    @classmethod
    def run(cls, files: list[PyIRFile], ctx: SymLinkContext) -> list[PyIRFile]:
        for ir in files:
            if ir.path is not None:
                ctx._all_paths.add(ir.path)

        ctx.ensure_module_index()
        cls._normalize_module_index_keys(ctx)

        path_by_module = cls._build_path_by_module(files, ctx)
        deps, exports_raw = cls._collect_deps_and_exports(files, ctx)

        defs_by_module = cls._collect_current_module_defs(files, ctx)
        exports: dict[str, set[str]] = {}
        for m, names in exports_raw.items():
            live = defs_by_module.get(m)
            if not live:
                continue

            kept = {n for n in names if n in live}
            if kept:
                exports[m] = kept

        ctx.project_deps = deps
        ctx.project_exports = exports
        ctx.project_path_by_module = path_by_module

        for ir in files:
            cls._process_file(ir, ctx, exports)

        return files

    @classmethod
    def _is_entity_or_weapon_file(cls, p: Path) -> bool:
        try:
            src_root = Py2GluaConfig.source.resolve()
            rel = p.resolve().relative_to(src_root)

        except Exception:
            rel = p

        parts = list(rel.parts)
        if parts and parts[0] == "lua":
            parts = parts[1:]

        return bool(parts) and parts[0] in cls._SKIP_AUTORUN_TOPDIRS

    @classmethod
    def _normalize_module_index_keys(cls, ctx: SymLinkContext) -> None:
        add: dict[Path, str] = {}
        for p, mod in list(ctx.module_name_by_path.items()):
            try:
                pr = p.resolve()
            except Exception:
                continue

            if pr not in ctx.module_name_by_path:
                add[pr] = mod

        if add:
            ctx.module_name_by_path.update(add)

    @classmethod
    def _build_path_by_module(
        cls, files: list[PyIRFile], ctx: SymLinkContext
    ) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for ir in files:
            if ir.path is None:
                continue

            if cls._is_entity_or_weapon_file(ir.path):
                continue

            mod = cls._module_of_path(ir.path, ctx)
            if mod:
                out[mod] = ir.path

        return out

    @classmethod
    def _collect_deps_and_exports(
        cls,
        files: list[PyIRFile],
        ctx: SymLinkContext,
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        deps: dict[str, set[str]] = {}
        exports: dict[str, set[str]] = {}

        for ir in files:
            if ir.path is None:
                continue

            if cls._is_entity_or_weapon_file(ir.path):
                continue

            use_mod = cls._module_of_path(ir.path, ctx)
            if not use_mod:
                continue

            for n in ir.walk():
                if not isinstance(n, PyIRSymLink):
                    continue

                if n.decl_file is None:
                    continue

                if cls._is_entity_or_weapon_file(n.decl_file):
                    continue

                decl_mod = cls._module_of_path(n.decl_file, ctx)
                if not decl_mod:
                    continue

                if decl_mod == use_mod:
                    continue

                deps.setdefault(use_mod, set()).add(decl_mod)
                exports.setdefault(decl_mod, set()).add(n.name)

        return deps, exports

    @classmethod
    def _collect_current_module_defs(
        cls, files: list[PyIRFile], ctx: SymLinkContext
    ) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}

        for ir in files:
            if ir.path is None:
                continue

            if cls._is_entity_or_weapon_file(ir.path):
                continue

            mod = cls._module_of_path(ir.path, ctx)
            if not mod:
                continue

            import_binds = cls._import_bind_map(ir)
            defs = cls._collect_defs_in_block(ir.body, stop_at_boundaries=True)
            defs = {n for n in defs if n not in import_binds}
            out[mod] = defs

        return out

    @classmethod
    def _process_file(
        cls,
        ir: PyIRFile,
        ctx: SymLinkContext,
        project_exports: dict[str, set[str]],
    ) -> None:
        if ir.path is None:
            return

        cur_mod = cls._module_of_path(ir.path, ctx)
        ns = (Py2GluaConfig.namespace or "").strip()
        import_binds = cls._import_bind_map(ir)

        cls._place_locals_project_style(ir, import_binds=import_binds)

        ir.body = [
            cls._rw_stmt(n, ctx, cur_mod, ir.path, import_binds, ns) for n in ir.body
        ]

        if cur_mod and ns:
            exported = sorted(project_exports.get(cur_mod, set()))
            if exported:
                ns_mod_parts = cls._ns_module_parts(cur_mod)
                ir.body = cls._append_exports(ir.body, ns, ns_mod_parts, exported)

    @classmethod
    def _place_locals_project_style(
        cls, ir: PyIRFile, *, import_binds: dict[str, tuple[str, ...]]
    ) -> None:
        module_locals = cls._collect_defs_in_block(ir.body, stop_at_boundaries=True)
        module_locals.difference_update(import_binds.keys())
        ir.body = cls._place_locals_in_block(
            ir.body, local_names=module_locals, params=set()
        )

        for st in ir.body:
            cls._place_locals_in_scopes_recursive(st)

    @classmethod
    def _place_locals_in_scopes_recursive(cls, node: PyIRNode) -> None:
        if isinstance(node, PyIRFunctionDef):
            fn_locals = cls._collect_defs_in_block(node.body, stop_at_boundaries=True)
            fn_params = set(node.signature.keys())
            if node.vararg is not None:
                fn_params.add(node.vararg)
            if node.kwarg is not None:
                fn_params.add(node.kwarg)
            fn_locals.difference_update(fn_params)
            node.body = cls._place_locals_in_block(
                node.body, local_names=fn_locals, params=fn_params
            )
            for st in node.body:
                cls._place_locals_in_scopes_recursive(st)
            return

        if isinstance(node, PyIRFunctionExpr):
            fn_locals = cls._collect_defs_in_block(node.body, stop_at_boundaries=True)
            fn_params = set(node.signature.keys())
            if node.vararg is not None:
                fn_params.add(node.vararg)
            if node.kwarg is not None:
                fn_params.add(node.kwarg)
            fn_locals.difference_update(fn_params)
            node.body = cls._place_locals_in_block(
                node.body, local_names=fn_locals, params=fn_params
            )
            for st in node.body:
                cls._place_locals_in_scopes_recursive(st)
            return

        if isinstance(node, PyIRClassDef):
            for st in node.body:
                cls._place_locals_in_scopes_recursive(st)
            return

        if not is_dataclass(node):
            return

        for f in fields(node):
            v = getattr(node, f.name)
            if isinstance(v, PyIRNode):
                cls._place_locals_in_scopes_recursive(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, PyIRNode):
                        cls._place_locals_in_scopes_recursive(x)
            elif isinstance(v, dict):
                for x in v.values():
                    if isinstance(x, PyIRNode):
                        cls._place_locals_in_scopes_recursive(x)

    @classmethod
    def _place_locals_in_block(
        cls,
        body: list[PyIRNode],
        *,
        local_names: set[str],
        params: set[str],
    ) -> list[PyIRNode]:
        if not local_names:
            return body

        local_names = {n for n in local_names if cls._is_simple_lua_name(n)}

        declared: set[str] = {
            st.name
            for st in body
            if isinstance(st, PyIRVarCreate) and cls._is_simple_lua_name(st.name)
        }
        needed = {n for n in local_names if n not in declared and n not in params}
        if not needed:
            return body

        first_at: dict[str, int] = {}

        for i, st in enumerate(body):
            touches = cls._touch_names_in_stmt(st)
            hits = (touches & needed) - params
            for n in hits:
                if n not in first_at:
                    first_at[n] = i
            if len(first_at) == len(needed):
                break

        if not first_at:
            return body

        by_index: dict[int, list[str]] = {}
        for n, idx in first_at.items():
            by_index.setdefault(idx, []).append(n)

        out: list[PyIRNode] = []
        for i, st in enumerate(body):
            names = by_index.get(i)
            if names:
                for n in sorted(names):
                    out.append(cls._mk_local_create(name=n))
            out.append(st)

        return out

    @classmethod
    def _touch_names_in_stmt(cls, node: PyIRNode) -> set[str]:
        out: set[str] = set()

        def add(name: str) -> None:
            if cls._is_simple_lua_name(name):
                out.add(name)

        def walk(n: PyIRNode, *, nested_scope: bool) -> None:
            if isinstance(n, PyIRSymLink):
                if (not nested_scope) or (n.hops > 0):
                    add(n.name)
                return

            if isinstance(n, PyIRVarUse):
                if not nested_scope:
                    add(n.name)
                return

            if isinstance(n, PyIRVarCreate):
                if not nested_scope:
                    add(n.name)
                return

            if isinstance(n, PyIRFunctionDef):
                if not nested_scope:
                    add(n.name)

                for d in n.decorators:
                    walk(d, nested_scope=nested_scope)

                for _arg, (_ann, default) in n.signature.items():
                    if default is not None:
                        walk(default, nested_scope=nested_scope)

                for b in n.body:
                    walk(b, nested_scope=True)
                return

            if isinstance(n, PyIRFunctionExpr):
                for _arg, (_ann, default) in n.signature.items():
                    if default is not None:
                        walk(default, nested_scope=nested_scope)
                for b in n.body:
                    walk(b, nested_scope=True)
                return

            if isinstance(n, PyIRClassDef):
                if not nested_scope:
                    add(n.name)

                for b in n.bases:
                    walk(b, nested_scope=nested_scope)
                for d in n.decorators:
                    walk(d, nested_scope=nested_scope)
                for b in n.body:
                    walk(b, nested_scope=True)
                return

            if not is_dataclass(n):
                return

            for f in fields(n):
                v = getattr(n, f.name)
                if isinstance(v, PyIRNode):
                    walk(v, nested_scope=nested_scope)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, PyIRNode):
                            walk(x, nested_scope=nested_scope)
                elif isinstance(v, dict):
                    for x in v.values():
                        if isinstance(x, PyIRNode):
                            walk(x, nested_scope=nested_scope)

        walk(node, nested_scope=False)
        return out

    @staticmethod
    def _mk_local_create(*, name: str) -> PyIRVarCreate:
        return PyIRVarCreate(
            line=None,
            offset=None,
            name=name,
            storage=PyIRVarStorage.LOCAL,
        )

    @classmethod
    def _append_exports(
        cls,
        body: list[PyIRNode],
        ns: str,
        ns_mod_parts: tuple[str, ...],
        exported_names_sorted: list[str],
    ) -> list[PyIRNode]:
        if not exported_names_sorted:
            return body

        out = list(body)
        ensure_lines = cls._emit_ensure_module_table(ns, ns_mod_parts)
        if ensure_lines:
            out.append(
                PyIREmitExpr(line=None, offset=None, kind=PyIREmitKind.RAW, name="")
            )
            out.extend(ensure_lines)

        for name in exported_names_sorted:
            lhs = cls._attr_chain(
                base=PyIRVarUse(line=None, offset=None, name=ns),
                parts=(*ns_mod_parts, name),
                line=None,
                offset=None,
            )
            out.append(
                PyIRAssign(
                    line=None,
                    offset=None,
                    targets=[lhs],
                    value=PyIRVarUse(line=None, offset=None, name=name),
                )
            )

        return out

    @classmethod
    def _emit_ensure_module_table(
        cls, ns: str, ns_mod_parts: tuple[str, ...]
    ) -> list[PyIRNode]:
        if not ns_mod_parts:
            return []

        stmts: list[str] = []
        for i in range(len(ns_mod_parts)):
            chain = ".".join([ns, *ns_mod_parts[: i + 1]])
            stmts.append(f"{chain} = {chain} or {{}}")

        out: list[PyIRNode] = []
        cur = ""
        for s in stmts:
            if not cur:
                cur = s
                continue
            cand = cur + "; " + s
            if len(cand) > cls._MAX_RAW_LINE:
                out.append(
                    PyIREmitExpr(
                        line=None, offset=None, kind=PyIREmitKind.RAW, name=cur
                    )
                )
                cur = s
            else:
                cur = cand

        if cur:
            out.append(
                PyIREmitExpr(line=None, offset=None, kind=PyIREmitKind.RAW, name=cur)
            )
        return out

    @classmethod
    def _rw_stmt(
        cls,
        node: PyIRNode,
        ctx: SymLinkContext,
        cur_mod: str | None,
        cur_file: Path,
        import_binds: dict[str, tuple[str, ...]],
        ns: str,
    ) -> PyIRNode:
        if isinstance(node, PyIRVarUse):
            imported = import_binds.get(node.name)
            if imported and ns:
                return cls._attr_chain(
                    base=PyIRVarUse(line=node.line, offset=node.offset, name=ns),
                    parts=imported,
                    line=node.line,
                    offset=node.offset,
                )
            return node

        if isinstance(node, PyIRSymLink):
            return cls._resolve_symlink(node, ctx, cur_mod, cur_file, import_binds, ns)

        if isinstance(node, PyIRAttribute):
            node.value = cls._rw_stmt(
                node.value, ctx, cur_mod, cur_file, import_binds, ns
            )
            return node

        if not is_dataclass(node):
            return node

        for f in fields(node):
            v = getattr(node, f.name)
            nv = cls._rw_value(v, ctx, cur_mod, cur_file, import_binds, ns)
            if nv is not v:
                setattr(node, f.name, nv)

        return node

    @classmethod
    def _rw_value(
        cls,
        v,
        ctx: SymLinkContext,
        cur_mod: str | None,
        cur_file: Path,
        import_binds: dict[str, tuple[str, ...]],
        ns: str,
    ):
        if isinstance(v, PyIRNode):
            return cls._rw_stmt(v, ctx, cur_mod, cur_file, import_binds, ns)

        if isinstance(v, list):
            changed = False
            out_list: list[object] = []
            for x in v:
                nx = cls._rw_value(x, ctx, cur_mod, cur_file, import_binds, ns)
                changed = changed or (nx is not x)
                out_list.append(nx)
            return out_list if changed else v

        if isinstance(v, dict):
            changed = False
            out_dict: dict[object, object] = {}
            for k, x in v.items():
                nk = cls._rw_value(k, ctx, cur_mod, cur_file, import_binds, ns)
                nx = cls._rw_value(x, ctx, cur_mod, cur_file, import_binds, ns)
                changed = changed or (nk is not k) or (nx is not x)
                out_dict[nk] = nx
            return out_dict if changed else v

        if isinstance(v, tuple):
            changed = False
            out_tuple: list[object] = []
            for x in v:
                nx = cls._rw_value(x, ctx, cur_mod, cur_file, import_binds, ns)
                changed = changed or (nx is not x)
                out_tuple.append(nx)
            return tuple(out_tuple) if changed else v

        return v

    @classmethod
    def _resolve_symlink(
        cls,
        n: PyIRSymLink,
        ctx: SymLinkContext,
        cur_mod: str | None,
        cur_file: Path,
        import_binds: dict[str, tuple[str, ...]],
        ns: str,
    ) -> PyIRNode:
        imported = import_binds.get(n.name)
        if imported and ns:
            return cls._attr_chain(
                base=PyIRVarUse(line=n.line, offset=n.offset, name=ns),
                parts=imported,
                line=n.line,
                offset=n.offset,
            )

        if n.decl_file is None or not ns:
            return PyIRVarUse(line=n.line, offset=n.offset, name=n.name)

        decl_mod = cls._module_of_path(n.decl_file, ctx)
        if decl_mod is None or cur_mod is None or decl_mod == cur_mod:
            return PyIRVarUse(line=n.line, offset=n.offset, name=n.name)

        parts = cls._ns_module_parts(decl_mod)
        return cls._attr_chain(
            base=PyIRVarUse(line=n.line, offset=n.offset, name=ns),
            parts=(*parts, n.name),
            line=n.line,
            offset=n.offset,
        )

    @classmethod
    def _collect_defs_in_block(
        cls, body: list[PyIRNode], *, stop_at_boundaries: bool
    ) -> set[str]:
        out: set[str] = set()

        def add_name(name: str) -> None:
            if cls._is_simple_lua_name(name):
                out.add(name)

        def add_simple_target(t: PyIRNode) -> None:
            if isinstance(t, PyIRVarUse):
                add_name(t.name)
                return
            if isinstance(t, PyIRSymLink) and t.is_store:
                add_name(t.name)
                return

        def walk(node: PyIRNode) -> None:
            if isinstance(node, PyIRVarCreate):
                add_name(node.name)
                return

            if isinstance(node, PyIRFunctionDef):
                add_name(node.name)
                if stop_at_boundaries:
                    return
                for b in node.body:
                    walk(b)
                return

            if isinstance(node, PyIRFunctionExpr):
                if stop_at_boundaries:
                    return
                for b in node.body:
                    walk(b)
                return

            if isinstance(node, PyIRClassDef):
                add_name(node.name)
                if stop_at_boundaries:
                    return
                for b in node.body:
                    walk(b)
                return

            if isinstance(node, PyIRAssign):
                for t in node.targets:
                    add_simple_target(t)
                walk(node.value)
                return

            if isinstance(node, PyIRAugAssign):
                add_simple_target(node.target)
                walk(node.value)
                return

            if isinstance(node, PyIRFor):
                add_simple_target(node.target)
                walk(node.iter)
                for b in node.body:
                    walk(b)
                return

            if isinstance(node, PyIRWithItem):
                if node.optional_vars is not None:
                    add_simple_target(node.optional_vars)
                walk(node.context_expr)
                return

            if isinstance(node, PyIRWith):
                for it in node.items:
                    walk(it)
                for b in node.body:
                    walk(b)
                return

            if isinstance(node, PyIRIf):
                walk(node.test)
                for b in node.body:
                    walk(b)
                for b in node.orelse:
                    walk(b)
                return

            if isinstance(node, PyIRWhile):
                walk(node.test)
                for b in node.body:
                    walk(b)
                return

            if isinstance(node, PyIRDo):
                for b in node.body:
                    walk(b)
                return

            if isinstance(node, PyIRCall):
                walk(node.func)
                for a in node.args_p:
                    walk(a)
                for a in node.args_kw.values():
                    walk(a)
                return

            if isinstance(node, PyIRAttribute):
                walk(node.value)
                return

            if isinstance(node, (PyIRImport, PyIRDecorator, PyIRComment)):
                return

            if is_dataclass(node):
                for f in fields(node):
                    v = getattr(node, f.name)
                    if isinstance(v, PyIRNode):
                        walk(v)
                    elif isinstance(v, list):
                        for x in v:
                            if isinstance(x, PyIRNode):
                                walk(x)
                    elif isinstance(v, dict):
                        for x in v.values():
                            if isinstance(x, PyIRNode):
                                walk(x)

        for st in body:
            walk(st)

        return out

    @staticmethod
    def _import_bind_map(ir: PyIRFile) -> dict[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = {}
        for n in ir.body:
            if not isinstance(n, PyIRImport):
                continue
            if n.if_from:
                continue

            mods = tuple(n.modules or ())
            if n.names:
                for spec in n.names:
                    if isinstance(spec, tuple):
                        src, alias = spec
                        src_parts = tuple(p for p in src.split(".") if p)
                        if not src_parts:
                            continue
                        bound = alias or src_parts[0]
                        out[bound] = src_parts if alias else (src_parts[0],)
                    else:
                        src = spec
                        src_parts = tuple(p for p in src.split(".") if p)
                        if not src_parts:
                            continue
                        out[src_parts[0]] = (src_parts[0],)
                continue

            if mods:
                out[mods[0]] = (mods[0],)

        return out

    @staticmethod
    def _ns_module_parts(mod_name: str) -> tuple[str, ...]:
        parts = [p for p in mod_name.split(".") if p]
        try:
            src_root_name = Py2GluaConfig.source.resolve().name
        except Exception:
            src_root_name = ""

        if parts and src_root_name and parts[0] == src_root_name:
            parts = parts[1:]
        return tuple(parts)

    @staticmethod
    def _attr_chain(
        *,
        base: PyIRNode,
        parts: tuple[str, ...] | list[str],
        line: int | None = None,
        offset: int | None = None,
    ) -> PyIRNode:
        cur: PyIRNode = base
        for p in parts:
            cur = PyIRAttribute(line=line, offset=offset, value=cur, attr=p)
        return cur

    @staticmethod
    def _module_of_path(p: Path, ctx: SymLinkContext) -> str | None:
        hit = ctx.module_name_by_path.get(p)
        if hit is not None:
            return hit
        try:
            pr = p.resolve()
        except Exception:
            return None
        return ctx.module_name_by_path.get(pr)

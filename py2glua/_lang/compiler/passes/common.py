from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from ...py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRDecorator,
    PyIRFile,
    PyIRImport,
    PyIRNode,
    PyIRVarUse,
)
from .analysis.symlinks import PyIRSymLink, SymLinkContext

_CORE_COMPILER_DIRECTIVE_MODULES: tuple[str, ...] = (
    "py2glua.glua.core",
    "py2glua.glua.core.compiler_directive",
)

_IMPORTED_LOCAL_NAMES_CACHE: dict[
    tuple[
        str, str, str | None, tuple[tuple[tuple[str, ...], tuple[str, ...], bool], ...]
    ],
    set[str],
] = {}


def canon_path(p: Path | None) -> Path | None:
    if p is None:
        return None
    try:
        return p.resolve()
    except Exception:
        return p


def iter_imported_names(
    names: Sequence[str | tuple[str, str]] | None,
) -> Iterator[tuple[str, str]]:
    if not names:
        return
    for n in names:
        if isinstance(n, tuple):
            src, alias = n
            yield src, alias
        else:
            yield n, n


def decorator_call_parts(
    d: PyIRDecorator,
) -> tuple[PyIRNode, list[PyIRNode], dict[str, PyIRNode]]:
    if isinstance(d.exper, PyIRCall):
        c = d.exper
        return c.func, list(c.args_p or []), dict(c.args_kw or {})
    return d.exper, list(d.args_p or []), dict(d.args_kw or {})


def unwrap_decorator_attribute(d: PyIRDecorator) -> PyIRAttribute | None:
    fn_expr, _args_p, _args_kw = decorator_call_parts(d)
    if isinstance(fn_expr, PyIRAttribute):
        return fn_expr
    return None


def collect_decl_symid_cache(
    ctx: SymLinkContext,
    *,
    cache_field: str,
) -> dict[tuple[Path, int, str], int]:
    cached = getattr(ctx, cache_field, None)
    if isinstance(cached, dict):
        return cached

    out: dict[tuple[Path, int, str], int] = {}
    for sym in ctx.symbols.values():
        if sym.decl_file is None or sym.decl_line is None:
            continue
        p = canon_path(sym.decl_file)
        if p is None:
            continue
        out[(p, sym.decl_line, sym.name)] = int(sym.id.value)

    setattr(ctx, cache_field, out)
    return out


def collect_exported_symbol_ids(ctx: SymLinkContext, symbol_name: str) -> set[int]:
    ids: set[int] = set()

    try:
        ctx.ensure_module_index()
        module_names = set(ctx.module_name_by_path.values())
    except Exception:
        return ids

    for mn in module_names:
        try:
            sym = ctx.get_exported_symbol(mn, symbol_name)
        except Exception:
            sym = None

        if sym is None:
            continue

        try:
            ids.add(int(sym.id.value))
        except Exception:
            continue

    return ids


def collect_imported_local_names(
    ir: PyIRFile,
    *,
    imported_name: str,
    module_substring: str | None = None,
) -> set[str]:
    file_key = str(canon_path(ir.path) or ir.path or "<none>")
    imports_sig: list[tuple[tuple[str, ...], tuple[str, ...], bool]] = []
    for n in ir.body:
        if not isinstance(n, PyIRImport):
            continue
        names_sig: list[str] = []
        for src, bound in iter_imported_names(n.names):
            names_sig.append(f"{src}:{bound}")
        imports_sig.append((tuple(n.modules or ()), tuple(names_sig), bool(n.if_from)))

    cache_key = (
        file_key,
        imported_name,
        module_substring,
        tuple(imports_sig),
    )
    cached = _IMPORTED_LOCAL_NAMES_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)

    names: set[str] = {imported_name}

    for n in ir.body:
        if not isinstance(n, PyIRImport):
            continue
        if not n.if_from:
            continue

        if module_substring is not None:
            mod = ".".join(n.modules)
            if module_substring not in mod:
                continue

        for src, bound in iter_imported_names(n.names):
            if src == imported_name and bound:
                names.add(bound)

    _IMPORTED_LOCAL_NAMES_CACHE[cache_key] = set(names)
    return names


def collect_symlink_ids_by_name(ir: PyIRFile, name: str) -> set[int]:
    out: set[int] = set()
    for n in ir.walk():
        if isinstance(n, PyIRSymLink) and n.name == name:
            out.add(int(n.symbol_id))
    return out


def collect_local_imported_symbol_ids(
    ir: PyIRFile,
    *,
    ctx: SymLinkContext,
    current_module: str,
    imported_name: str,
    allowed_modules: Sequence[str] | None = None,
    module_substring: str | None = None,
) -> set[int]:
    out: set[int] = set()
    allowed_set = set(allowed_modules or ())

    def local_symbol_id(name: str) -> int | None:
        sym = ctx.get_exported_symbol(current_module, name)
        return int(sym.id.value) if sym is not None else None

    for n in ir.walk():
        if not isinstance(n, PyIRImport):
            continue
        if not n.if_from:
            continue
        module_name = ".".join(n.modules or ())
        if allowed_set and module_name not in allowed_set:
            continue
        if module_substring is not None and module_substring not in module_name:
            continue

        for src, bound in iter_imported_names(n.names):
            if src != imported_name:
                continue
            sid = local_symbol_id(bound)
            if sid is not None:
                out.add(sid)

    return out


def collect_local_class_symbol_ids(
    ir: PyIRFile,
    *,
    class_name: str,
    file_path: Path,
    symid_by_decl: dict[tuple[Path, int, str], int],
) -> set[int]:
    out: set[int] = set()
    for n in ir.walk():
        if isinstance(n, PyIRClassDef) and n.name == class_name and n.line is not None:
            sid = symid_by_decl.get((file_path, n.line, n.name))
            if sid is not None:
                out.add(int(sid))
            break
    return out


def is_named_ref(
    n: PyIRNode, names: set[str], *, allow_attr_chain: bool = False
) -> bool:
    if isinstance(n, PyIRVarUse):
        return n.name in names

    if isinstance(n, PyIRSymLink):
        return n.name in names

    if allow_attr_chain and isinstance(n, PyIRAttribute):
        if n.attr in names:
            return True
        return is_named_ref(n.value, names, allow_attr_chain=True)

    return False


def collect_symbol_ids_in_modules(
    ctx: SymLinkContext,
    *,
    symbol_name: str,
    modules: Sequence[str],
) -> set[int]:
    ids: set[int] = set()
    try:
        ctx.ensure_module_index()
    except Exception:
        return ids

    for module_name in modules:
        try:
            sym = ctx.get_exported_symbol(module_name, symbol_name)
        except Exception:
            sym = None

        if sym is None:
            continue

        try:
            ids.add(int(sym.id.value))
        except Exception:
            continue

    return ids


def collect_core_compiler_directive_symbol_ids(ctx: SymLinkContext) -> set[int]:
    return collect_symbol_ids_in_modules(
        ctx,
        symbol_name="CompilerDirective",
        modules=_CORE_COMPILER_DIRECTIVE_MODULES,
    )


def collect_core_compiler_directive_local_symbol_ids(
    ir: PyIRFile,
    *,
    ctx: SymLinkContext,
    current_module: str,
) -> set[int]:
    ids = collect_core_compiler_directive_symbol_ids(ctx)
    ids |= collect_local_imported_symbol_ids(
        ir,
        ctx=ctx,
        current_module=current_module,
        imported_name="CompilerDirective",
        allowed_modules=_CORE_COMPILER_DIRECTIVE_MODULES,
    )
    return ids


def unwrap_calls(expr: PyIRNode) -> PyIRNode:
    while isinstance(expr, PyIRCall):
        expr = expr.func
    return expr


def is_attr_on_symbol_ids(
    expr: PyIRNode,
    *,
    attr: str,
    symbol_ids: set[int],
    ctx: SymLinkContext | None = None,
) -> bool:
    base_expr = unwrap_calls(expr)
    if not isinstance(base_expr, PyIRAttribute):
        return False
    if base_expr.attr != attr:
        return False
    base = base_expr.value
    if isinstance(base, PyIRSymLink):
        return int(base.symbol_id) in symbol_ids
    if ctx is None:
        return False

    resolved = resolve_symbol_ids_by_attr_chain(ctx, base)
    return any(sid in symbol_ids for sid in resolved)


def is_expr_on_symbol_ids(
    expr: PyIRNode,
    *,
    symbol_ids: set[int],
    ctx: SymLinkContext | None = None,
) -> bool:
    if isinstance(expr, PyIRSymLink):
        return int(expr.symbol_id) in symbol_ids
    if ctx is None:
        return False

    resolved = resolve_symbol_ids_by_attr_chain(ctx, expr)
    return any(sid in symbol_ids for sid in resolved)


def attr_chain_parts(expr: PyIRNode) -> list[str] | None:
    attrs_rev: list[str] = []
    cur = expr
    while isinstance(cur, PyIRAttribute):
        attrs_rev.append(cur.attr)
        cur = cur.value
    if isinstance(cur, (PyIRVarUse, PyIRSymLink)):
        root_name = cur.name
    else:
        return None
    attrs_rev.reverse()
    parts = [root_name, *attrs_rev]
    while len(parts) > 1 and parts[0] == parts[1]:
        parts = parts[1:]
    return parts


def resolve_symbol_ids_by_attr_chain(
    ctx: SymLinkContext,
    expr: PyIRNode,
) -> set[int]:
    if isinstance(expr, PyIRSymLink):
        return {int(expr.symbol_id)}

    parts = attr_chain_parts(expr)
    if not parts or len(parts) < 2:
        return set()

    out: set[int] = set()
    module_names = tuple(ctx.module_exports.keys())

    candidates = [parts]
    if len(parts) > 2:
        for i in range(1, len(parts) - 1):
            candidates.append(parts[i:])

    for cur_parts in candidates:
        for j in range(len(cur_parts) - 1, 0, -1):
            suffix = tuple(cur_parts[:j])
            export_name = cur_parts[j]
            for module_name in module_names:
                mod_parts = tuple(module_name.split("."))
                if len(mod_parts) < len(suffix):
                    continue
                if tuple(mod_parts[-len(suffix) :]) != suffix:
                    continue

                exports = ctx.module_exports.get(module_name)
                if not exports:
                    continue
                sid = exports.get(export_name)
                if sid is not None:
                    out.add(int(sid.value))

    if out:
        return out

    # Fallback for re-export style chains where module index does not include imports
    # as exports (for example: py2glua.glua.core.CompilerDirective).
    module_name = ".".join(parts[:-1])
    export_name = parts[-1]

    snake = []
    for i, ch in enumerate(export_name):
        if ch.isupper() and i > 0 and export_name[i - 1].islower():
            snake.append("_")
        snake.append(ch.lower())
    snake_name = "".join(snake)

    extra_modules = (
        module_name,
        f"{module_name}.{export_name}",
        f"{module_name}.{snake_name}",
    )
    for mod in extra_modules:
        try:
            sym = ctx.get_exported_symbol(mod, export_name)
        except Exception:
            sym = None
        if sym is not None:
            out.add(int(sym.id.value))

    return out

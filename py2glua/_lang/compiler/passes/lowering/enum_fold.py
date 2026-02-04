from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ....compiler.passes.analysis.symlinks import PyIRSymLink, SymLinkContext
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBinOP,
    PyIRBreak,
    PyIRCall,
    PyIRClassDef,
    PyIRComment,
    PyIRConstant,
    PyIRContinue,
    PyIRDecorator,
    PyIRDict,
    PyIRDictItem,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRList,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
    PyIRSet,
    PyIRSubscript,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarCreate,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)


@dataclass(frozen=True)
class _EnumFieldSpec:
    kind: str  # "global" | "bool" | "int" | "float" | "str"
    value: Any  # for "global": identifier string; for others: python literal


def _get_store(ctx: SymLinkContext) -> dict[int, dict[str, _EnumFieldSpec]]:
    store = getattr(ctx, "_gmod_special_enum_store", None)
    if store is None:
        store = {}
        setattr(ctx, "_gmod_special_enum_store", store)
    return store


def _iter_imported_names(
    names: list[str | tuple[str, str]],
) -> Iterable[tuple[str, str]]:
    for n in names:
        if isinstance(n, tuple):
            yield n[0], n[1]
        else:
            yield n, n


def _is_gmod_special_enum_decorator(d: PyIRDecorator) -> bool:
    exp = d.exper
    return isinstance(exp, PyIRAttribute) and exp.attr == "gmod_special_enum"


def _parse_fields_dict(node: PyIRNode) -> dict[str, _EnumFieldSpec]:
    if not isinstance(node, PyIRDict):
        raise ValueError("gmod_special_enum(fields=...) expects a dict")

    out: dict[str, _EnumFieldSpec] = {}

    for it in node.items:
        if not isinstance(it, PyIRDictItem):
            continue

        if not isinstance(it.key, PyIRConstant) or not isinstance(it.key.value, str):
            raise ValueError("gmod_special_enum fields keys must be string constants")

        field_name = it.key.value

        if not isinstance(it.value, PyIRTuple) or len(it.value.elements) != 2:
            raise ValueError(
                "gmod_special_enum fields values must be a 2-tuple: (kind, value)"
            )

        kind_node, val_node = it.value.elements

        if not isinstance(kind_node, PyIRConstant) or not isinstance(
            kind_node.value, str
        ):
            raise ValueError("gmod_special_enum field kind must be a string constant")

        if not isinstance(val_node, PyIRConstant):
            raise ValueError("gmod_special_enum field value must be a constant")

        out[field_name] = _EnumFieldSpec(kind=kind_node.value, value=val_node.value)

    return out


def _make_global(ref: PyIRNode, ident: str) -> PyIREmitExpr:
    return PyIREmitExpr(
        line=ref.line,
        offset=ref.offset,
        kind=PyIREmitKind.GLOBAL,
        name=ident,
        args_p=[],
        args_kw={},
    )


def _make_replacement(ref: PyIRNode, spec: _EnumFieldSpec) -> PyIRNode:
    kind = spec.kind
    val = spec.value

    if kind == "global":
        if not isinstance(val, str):
            raise ValueError(
                "gmod_special_enum global value must be a string identifier"
            )

        return _make_global(ref, val)

    if kind == "bool":
        if not isinstance(val, bool):
            raise ValueError("gmod_special_enum bool value must be a bool constant")

        return PyIRConstant(line=ref.line, offset=ref.offset, value=val)

    if kind in ("int", "float", "str"):
        return PyIRConstant(line=ref.line, offset=ref.offset, value=val)

    raise ValueError(f"Unsupported enum field kind: {kind!r}")


def _const_to_spec(value_node: PyIRNode) -> _EnumFieldSpec | None:
    if isinstance(value_node, PyIRConstant):
        v = value_node.value
        if isinstance(v, bool):
            return _EnumFieldSpec(kind="bool", value=v)

        if isinstance(v, int):
            return _EnumFieldSpec(kind="int", value=v)

        if isinstance(v, float):
            return _EnumFieldSpec(kind="float", value=v)

        if isinstance(v, str):
            return _EnumFieldSpec(kind="str", value=v)

        return None

    if isinstance(value_node, PyIREmitExpr) and value_node.kind == PyIREmitKind.GLOBAL:
        return _EnumFieldSpec(kind="global", value=value_node.name)

    return None


def _extract_target_name(ctx: SymLinkContext, t: PyIRNode) -> str | None:
    if isinstance(t, PyIRVarUse):
        return t.name

    if isinstance(t, PyIRSymLink):
        name = getattr(t, "name", None)
        if isinstance(name, str) and name:
            return name

        for attr in ("symbols_by_id", "symbol_by_id", "id_to_symbol", "symbols"):
            m = getattr(ctx, attr, None)
            if isinstance(m, dict):
                sym = m.get(t.symbol_id)
                if sym is not None:
                    n = getattr(sym, "name", None)
                    if isinstance(n, str) and n:
                        return n

    return None


def _collect_import_module_aliases(
    ir: PyIRFile,
    ctx: SymLinkContext,
    current_module: str,
) -> dict[int, str]:
    """
    Builds mapping: alias_symbol_id -> imported module name string.
    Needed for resolving `module.ClassName` into class symbol id.
    """

    def local_symbol_id(name: str) -> int | None:
        sym = ctx.get_exported_symbol(current_module, name)
        return sym.id.value if sym is not None else None

    out: dict[int, str] = {}

    for node in ir.walk():
        if not isinstance(node, PyIRImport):
            continue
        if node.if_from:
            continue

        mod = ".".join(node.modules) if node.modules else ""
        if not mod:
            continue

        if node.names:
            for _orig, bound in _iter_imported_names(node.names):
                sid = local_symbol_id(bound)
                if sid is not None:
                    out[sid] = mod

        else:
            bound = node.modules[-1] if node.modules else ""
            sid = local_symbol_id(bound)
            if sid is not None:
                out[sid] = mod

    return out


def _collect_enum_base_ids(
    ir: PyIRFile,
    ctx: SymLinkContext,
    current_module: str,
    import_alias_to_mod: dict[int, str],
) -> set[int]:
    """
    Returns symbol_ids of locally bound names that represent Enum bases (Enum/IntEnum).
    Supports both:
      from enum import Enum, IntEnum
      import enum; enum.Enum / enum.IntEnum
    """

    def local_symbol_id(name: str) -> int | None:
        sym = ctx.get_exported_symbol(current_module, name)
        return sym.id.value if sym is not None else None

    enum_base_ids: set[int] = set()

    for node in ir.walk():
        if not isinstance(node, PyIRImport):
            continue

        mod = ".".join(node.modules) if node.modules else ""
        if mod != "enum":
            continue

        if node.if_from:
            for orig, bound in _iter_imported_names(node.names):
                if orig in ("Enum", "IntEnum", "StrEnum"):
                    sid = local_symbol_id(bound)
                    if sid is not None:
                        enum_base_ids.add(sid)

    return enum_base_ids


def _is_python_enum_base(
    base: PyIRNode,
    *,
    enum_name_ids: set[int],
    import_alias_to_mod: dict[int, str],
) -> bool:
    """
    True if base expression represents Enum/IntEnum/StrEnum.
    Works for:
      Enum
      IntEnum
      enum.Enum
      enum.IntEnum
      <alias>.Enum  (where alias imports module "enum")
    """
    if isinstance(base, PyIRVarUse) and base.name in ("Enum", "IntEnum", "StrEnum"):
        return True

    if isinstance(base, PyIRSymLink) and base.symbol_id in enum_name_ids:
        return True

    if isinstance(base, PyIRAttribute) and base.attr in ("Enum", "IntEnum", "StrEnum"):
        v = base.value
        if isinstance(v, PyIRVarUse) and v.name == "enum":
            return True
        if isinstance(v, PyIRSymLink):
            mod = import_alias_to_mod.get(v.symbol_id)
            if mod == "enum":
                return True

    return False


def _class_is_python_enum(
    cls: PyIRClassDef,
    *,
    enum_name_ids: set[int],
    import_alias_to_mod: dict[int, str],
) -> bool:
    return any(
        _is_python_enum_base(
            b, enum_name_ids=enum_name_ids, import_alias_to_mod=import_alias_to_mod
        )
        for b in cls.bases
    )


def _resolve_enum_class_id(
    expr: PyIRNode,
    ctx: SymLinkContext,
    store: dict[int, dict[str, _EnumFieldSpec]],
    import_alias_to_mod: dict[int, str],
) -> int | None:
    """
    Resolves expressions that point to enum-class object:
      Realm
      module.Realm
    into class symbol_id, if present in store.
    """
    if isinstance(expr, PyIRSymLink) and not expr.is_store:
        return expr.symbol_id if expr.symbol_id in store else None

    if isinstance(expr, PyIRAttribute):
        if isinstance(expr.value, PyIRSymLink) and not expr.value.is_store:
            mod = import_alias_to_mod.get(expr.value.symbol_id)
            if mod:
                sym = ctx.get_exported_symbol(mod, expr.attr)
                if sym is not None and sym.id.value in store:
                    return sym.id.value

    return None


class CollectGmodSpecialEnumDeclsPass:
    """
    Collects mapping for:
      - classes with @gmod_special_enum(fields=...)
      - OR classes that inherit Python Enum/IntEnum (values from NAME = <const> in class body)
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        if ir.path is None:
            return ir

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        store = _get_store(ctx)

        import_alias_to_mod = _collect_import_module_aliases(ir, ctx, current_module)
        enum_name_ids = _collect_enum_base_ids(
            ir, ctx, current_module, import_alias_to_mod
        )

        for node in ir.walk():
            if not isinstance(node, PyIRClassDef):
                continue

            sym = ctx.get_exported_symbol(current_module, node.name)
            if sym is None:
                continue

            cls_id = sym.id.value

            dec: PyIRDecorator | None = None
            for d in node.decorators:
                if isinstance(d, PyIRDecorator) and _is_gmod_special_enum_decorator(d):
                    dec = d
                    break

            if dec is not None:
                fields_node = dec.args_kw.get("fields")
                if fields_node is None:
                    raise ValueError(
                        "gmod_special_enum missing required kw arg: fields"
                    )

                mapping = _parse_fields_dict(fields_node)
                store.setdefault(cls_id, {}).update(mapping)
                continue

            if not _class_is_python_enum(
                node,
                enum_name_ids=enum_name_ids,
                import_alias_to_mod=import_alias_to_mod,
            ):
                continue

            body_mapping: dict[str, _EnumFieldSpec] = {}

            for st in node.body:
                if not isinstance(st, PyIRAssign):
                    continue
                if len(st.targets) != 1:
                    continue

                t = st.targets[0]
                name = _extract_target_name(ctx, t)
                if not name:
                    continue

                spec = _const_to_spec(st.value)
                if spec is None:
                    continue

                body_mapping[name] = spec

            if body_mapping:
                store.setdefault(cls_id, {}).update(body_mapping)

        return ir


class FoldGmodSpecialEnumUsesPass:
    """
    Replaces enum member reads ONLY when the base resolves to a collected enum-class:
      Realm.SERVER            -> <const>
      module.Realm.SERVER     -> <const>
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        store: dict[int, dict[str, _EnumFieldSpec]] | None = getattr(
            ctx, "_gmod_special_enum_store", None
        )
        if not store:
            return ir

        if ir.path is None:
            return ir

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        import_alias_to_mod = _collect_import_module_aliases(ir, ctx, current_module)

        def rw(node: PyIRNode, *, store_ctx: bool) -> PyIRNode:
            if isinstance(node, PyIRAttribute):
                node.value = rw(node.value, store_ctx=False)

                if store_ctx:
                    return node

                base_cls_id = _resolve_enum_class_id(
                    node.value, ctx, store, import_alias_to_mod
                )
                if base_cls_id is not None:
                    spec = store[base_cls_id].get(node.attr)
                    if spec is not None:
                        return _make_replacement(node, spec)

                return node

            if isinstance(node, PyIRSymLink):
                return node

            if isinstance(node, PyIRCall):
                node.func = rw(node.func, store_ctx=False)
                node.args_p = [rw(a, store_ctx=False) for a in node.args_p]
                node.args_kw = {
                    k: rw(v, store_ctx=False) for k, v in node.args_kw.items()
                }
                return node

            if isinstance(
                node,
                (
                    PyIRConstant,
                    PyIRVarUse,
                    PyIRVarCreate,
                    PyIRImport,
                    PyIRBreak,
                    PyIRContinue,
                    PyIRPass,
                    PyIRComment,
                    PyIREmitExpr,
                ),
            ):
                return node

            if isinstance(node, PyIRUnaryOP):
                node.value = rw(node.value, store_ctx=False)
                return node

            if isinstance(node, PyIRBinOP):
                node.left = rw(node.left, store_ctx=False)
                node.right = rw(node.right, store_ctx=False)
                return node

            if isinstance(node, PyIRSubscript):
                node.value = rw(node.value, store_ctx=False)
                node.index = rw(node.index, store_ctx=False)
                return node

            if isinstance(node, PyIRList):
                node.elements = [rw(e, store_ctx=False) for e in node.elements]
                return node

            if isinstance(node, PyIRTuple):
                node.elements = [rw(e, store_ctx=False) for e in node.elements]
                return node

            if isinstance(node, PyIRSet):
                node.elements = [rw(e, store_ctx=False) for e in node.elements]
                return node

            if isinstance(node, PyIRDictItem):
                node.key = rw(node.key, store_ctx=False)
                node.value = rw(node.value, store_ctx=False)
                return node

            if isinstance(node, PyIRDict):
                node.items = [rw(i, store_ctx=False) for i in node.items]  # type: ignore[list-item]
                return node

            if isinstance(node, PyIRDecorator):
                node.exper = rw(node.exper, store_ctx=False)
                node.args_p = [rw(a, store_ctx=False) for a in node.args_p]
                node.args_kw = {
                    k: rw(v, store_ctx=False) for k, v in node.args_kw.items()
                }
                return node

            if isinstance(node, PyIRAssign):
                node.targets = [rw(t, store_ctx=True) for t in node.targets]
                node.value = rw(node.value, store_ctx=False)
                return node

            if isinstance(node, PyIRAugAssign):
                node.target = rw(node.target, store_ctx=True)
                node.value = rw(node.value, store_ctx=False)
                return node

            if isinstance(node, PyIRReturn):
                node.value = rw(node.value, store_ctx=False)
                return node

            if isinstance(node, PyIRIf):
                node.test = rw(node.test, store_ctx=False)
                node.body = [rw(s, store_ctx=False) for s in node.body]
                node.orelse = [rw(s, store_ctx=False) for s in node.orelse]
                return node

            if isinstance(node, PyIRWhile):
                node.test = rw(node.test, store_ctx=False)
                node.body = [rw(s, store_ctx=False) for s in node.body]
                return node

            if isinstance(node, PyIRFor):
                node.target = rw(node.target, store_ctx=True)
                node.iter = rw(node.iter, store_ctx=False)
                node.body = [rw(s, store_ctx=False) for s in node.body]
                return node

            if isinstance(node, PyIRWithItem):
                node.context_expr = rw(node.context_expr, store_ctx=False)
                if node.optional_vars is not None:
                    node.optional_vars = rw(node.optional_vars, store_ctx=True)
                return node

            if isinstance(node, PyIRWith):
                node.items = [rw(i, store_ctx=False) for i in node.items]  # type: ignore[list-item]
                node.body = [rw(s, store_ctx=False) for s in node.body]
                return node

            if isinstance(node, PyIRFunctionDef):
                node.decorators = [rw(d, store_ctx=False) for d in node.decorators]  # type: ignore[list-item]
                node.body = [rw(s, store_ctx=False) for s in node.body]
                return node

            if isinstance(node, PyIRClassDef):
                node.bases = [rw(b, store_ctx=False) for b in node.bases]
                node.decorators = [rw(d, store_ctx=False) for d in node.decorators]  # type: ignore[list-item]
                node.body = [rw(s, store_ctx=False) for s in node.body]
                return node

            return node

        ir.body = [rw(n, store_ctx=False) for n in ir.body]
        return ir

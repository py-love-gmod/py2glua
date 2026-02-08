from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ....._cli.compiler_exit import CompilerExit
from ....compiler.passes.analysis.symlinks import PyIRSymLink, SymLinkContext
from ....py.ir_dataclass import (
    PyIRAnnotatedAssign,
    PyIRAssign,
    PyIRAttribute,
    PyIRClassDef,
    PyIRConstant,
    PyIRDecorator,
    PyIRDict,
    PyIRDictItem,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRNode,
    PyIRVarUse,
)
from ..common import (
    collect_core_compiler_directive_local_symbol_ids,
    decorator_call_parts,
    is_expr_on_symbol_ids,
)
from ..rewrite_utils import rewrite_expr_with_store_default, rewrite_stmt_block


def _sid(x: Any) -> int | None:
    if isinstance(x, int):
        return x
    v = getattr(x, "value", None)
    if isinstance(v, int):
        return v
    return None


@dataclass(frozen=True)
class _EnumFieldSpec:
    kind: str
    value: Any


def _get_store(ctx: SymLinkContext) -> dict[int, dict[str, _EnumFieldSpec]]:
    return cast(dict[int, dict[str, _EnumFieldSpec]], ctx._gmod_special_enum_store)


def _is_gmod_special_enum_decorator(
    d: PyIRDecorator,
    *,
    core_cd_symids: set[int],
    ctx: SymLinkContext,
) -> bool:
    func, _, _ = decorator_call_parts(d)
    if not isinstance(func, PyIRAttribute):
        return False
    if func.attr != "gmod_special_enum":
        return False
    return is_expr_on_symbol_ids(
        func.value,
        symbol_ids=core_cd_symids,
        ctx=ctx,
    )


def _const_to_spec(node: PyIRNode) -> _EnumFieldSpec | None:
    if isinstance(node, PyIRConstant):
        v = node.value
        if isinstance(v, bool):
            return _EnumFieldSpec("bool", v)
        if isinstance(v, int) and not isinstance(v, bool):
            return _EnumFieldSpec("int", v)
        if isinstance(v, float):
            return _EnumFieldSpec("float", v)
        if isinstance(v, str):
            return _EnumFieldSpec("str", v)
        return None

    if isinstance(node, PyIREmitExpr) and node.kind == PyIREmitKind.GLOBAL:
        if isinstance(node.name, str) and node.name:
            return _EnumFieldSpec("global", node.name)

    return None


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
    if spec.kind == "global":
        return _make_global(ref, spec.value)
    return PyIRConstant(ref.line, ref.offset, spec.value)


def _dotted_parts(expr: PyIRNode) -> list[str] | None:
    if isinstance(expr, PyIRVarUse):
        return [expr.name] if expr.name else None

    if isinstance(expr, PyIRSymLink):
        n = getattr(expr, "name", None)
        return [n] if isinstance(n, str) and n else None

    if isinstance(expr, PyIREmitExpr) and expr.kind == PyIREmitKind.GLOBAL:
        return [expr.name] if expr.name else None

    if isinstance(expr, PyIRAttribute):
        base = _dotted_parts(expr.value)
        if base is None:
            return None
        return [*base, expr.attr]

    return None


def _is_python_enum_base(base: PyIRNode) -> bool:
    if isinstance(base, PyIRVarUse) and base.name in ("Enum", "IntEnum", "StrEnum"):
        return True

    if isinstance(base, PyIRSymLink):
        if base.name in ("Enum", "IntEnum", "StrEnum"):
            return True

    parts = _dotted_parts(base)
    if parts and len(parts) >= 2:
        return parts[-2] == "enum" and parts[-1] in ("Enum", "IntEnum", "StrEnum")

    return False


def _class_is_python_enum(cls: PyIRClassDef) -> bool:
    return any(_is_python_enum_base(b) for b in (cls.bases or []))


def _extract_target_name(ctx: SymLinkContext, t: PyIRNode) -> str | None:
    _ = ctx
    if isinstance(t, PyIRVarUse):
        return t.name

    if isinstance(t, PyIRSymLink):
        if isinstance(t.name, str) and t.name:
            return t.name

    return None


def _resolve_enum_class_id(
    expr: PyIRNode,
    *,
    ctx: SymLinkContext,
    store: dict[int, dict[str, _EnumFieldSpec]],
    current_module: str,
) -> int | None:
    if isinstance(expr, PyIRSymLink) and not expr.is_store:
        sid = _sid(expr.symbol_id)
        return sid if sid in store else None

    parts = _dotted_parts(expr)
    if not parts:
        return None

    if len(parts) == 1:
        sym = ctx.get_exported_symbol(current_module, parts[0])
    else:
        sym = ctx.get_exported_symbol(".".join(parts[:-1]), parts[-1])

    if sym is None:
        return None

    cid = _sid(sym.id)
    return cid if cid in store else None


class CollectGmodSpecialEnumDeclsPass:
    """
    Collects enum field mappings from:
      - @gmod_special_enum(fields=...)
      - Python Enum / IntEnum / StrEnum classes
        including annotated assignments
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        if ir.path is None:
            return ir

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        core_cd_symids = collect_core_compiler_directive_local_symbol_ids(
            ir,
            ctx=ctx,
            current_module=current_module,
        )
        store = _get_store(ctx)

        for node in ir.walk():
            if not isinstance(node, PyIRClassDef):
                continue

            sym = ctx.get_exported_symbol(current_module, node.name)
            if sym is None:
                continue

            cls_id = _sid(sym.id)
            if cls_id is None:
                continue

            for d in node.decorators or []:
                if not _is_gmod_special_enum_decorator(
                    d,
                    core_cd_symids=core_cd_symids,
                    ctx=ctx,
                ):
                    continue

                _, _, kw = decorator_call_parts(d)
                fields = kw.get("fields")
                if not isinstance(fields, PyIRDict):
                    CompilerExit.user_error_node(
                        "gmod_special_enum(fields=...) ожидает словарь",
                        path=ir.path,
                        node=d,
                    )

                mapping: dict[str, _EnumFieldSpec] = {}
                for it in fields.items:
                    if not isinstance(it, PyIRDictItem):
                        continue

                    if not isinstance(it.key, PyIRConstant):
                        continue
                    if not isinstance(it.key.value, str):
                        continue

                    spec = _const_to_spec(it.value)
                    if spec:
                        mapping[it.key.value] = spec

                store.setdefault(cls_id, {}).update(mapping)
                break

            if not _class_is_python_enum(node):
                continue

            body_map: dict[str, _EnumFieldSpec] = {}

            for st in node.body or []:
                if isinstance(st, PyIRAssign):
                    if len(st.targets) != 1:
                        continue
                    name = _extract_target_name(ctx, st.targets[0])
                    if not name:
                        continue
                    spec = _const_to_spec(st.value)
                    if spec:
                        body_map[name] = spec

                elif isinstance(st, PyIRAnnotatedAssign):
                    name = st.name
                    spec = _const_to_spec(st.value)
                    if spec:
                        body_map[name] = spec

            if body_map:
                store.setdefault(cls_id, {}).update(body_map)

        return ir


class FoldGmodSpecialEnumUsesPass:
    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        store = _get_store(ctx)
        if not store or ir.path is None:
            return ir

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(node: PyIRNode, store_ctx: bool) -> PyIRNode:
            if isinstance(node, PyIRAttribute):
                node.value = rw_expr(node.value, store_ctx=False)

                if not store_ctx:
                    cid = _resolve_enum_class_id(
                        node.value,
                        ctx=ctx,
                        store=store,
                        current_module=current_module,
                    )
                    if cid is not None:
                        spec = store.get(cid, {}).get(node.attr)
                        if spec:
                            return _make_replacement(node, spec)

                return node

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body or [])
        return ir

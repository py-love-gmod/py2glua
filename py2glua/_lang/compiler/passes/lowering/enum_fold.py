from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ....._cli.compiler_exit import CompilerExit
from ....compiler.passes.analysis.symlinks import PyIRSymLink, SymLinkContext
from ....py.ir_dataclass import (
    PyIRAnnotatedAssign,
    PyIRAssign,
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRDecorator,
    PyIRDict,
    PyIRDictItem,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRImport,
    PyIRNode,
    PyIRVarUse,
)
from ..common import (
    collect_core_compiler_directive_local_symbol_ids,
    collect_local_imported_symbol_ids,
    collect_symbol_ids_in_modules,
    decorator_call_parts,
    is_expr_on_symbol_ids,
    iter_imported_names,
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


_ENUM_KIND_ENUM = "Enum"
_ENUM_KIND_INT = "IntEnum"
_ENUM_KIND_STR = "StrEnum"

_ENUM_MODULES: tuple[str, ...] = ("enum",)
_ENUM_BASE_NAMES: tuple[str, ...] = (
    _ENUM_KIND_ENUM,
    _ENUM_KIND_INT,
    _ENUM_KIND_STR,
)


def _get_store(ctx: SymLinkContext) -> dict[int, dict[str, _EnumFieldSpec]]:
    return cast(dict[int, dict[str, _EnumFieldSpec]], ctx._gmod_special_enum_store)


def _get_nested_class_ids(ctx: SymLinkContext) -> dict[tuple[int, str], int]:
    return cast(dict[tuple[int, str], int], ctx._gmod_special_enum_nested_ids)


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


def _enum_kind_from_base(
    base: PyIRNode,
    *,
    enum_symbol_kind_by_id: dict[int, str],
) -> str | None:
    if isinstance(base, PyIRSymLink):
        sid = _sid(base.symbol_id)
        if sid is not None and sid in enum_symbol_kind_by_id:
            return enum_symbol_kind_by_id[sid]
        if base.name in _ENUM_BASE_NAMES:
            return str(base.name)

    if isinstance(base, PyIRVarUse) and base.name in _ENUM_BASE_NAMES:
        return str(base.name)

    parts = _dotted_parts(base)
    if parts and len(parts) >= 2:
        tail = parts[-1]
        if parts[-2] == "enum" and tail in _ENUM_BASE_NAMES:
            return tail

    return None


def _detect_enum_kind(
    cls: PyIRClassDef,
    *,
    enum_symbol_kind_by_id: dict[int, str],
) -> str | None:
    for b in cls.bases or []:
        kind = _enum_kind_from_base(
            b,
            enum_symbol_kind_by_id=enum_symbol_kind_by_id,
        )
        if kind is not None:
            return kind
    return None


def _is_auto_func_expr(
    func: PyIRNode,
    *,
    auto_symbol_ids: set[int],
    local_auto_names: set[str],
    ctx: SymLinkContext,
) -> bool:
    if isinstance(func, PyIRSymLink):
        sid = _sid(func.symbol_id)
        if sid is not None and sid in auto_symbol_ids:
            return True
        return func.name in local_auto_names

    if isinstance(func, PyIRVarUse):
        return func.name in local_auto_names

    if isinstance(func, PyIRAttribute):
        if func.attr != "auto":
            return False
        if is_expr_on_symbol_ids(func, symbol_ids=auto_symbol_ids, ctx=ctx):
            return True
        parts = _dotted_parts(func)
        return bool(
            parts and len(parts) >= 2 and parts[-2] == "enum" and parts[-1] == "auto"
        )

    return False


def _is_auto_call(
    node: PyIRNode,
    *,
    auto_symbol_ids: set[int],
    local_auto_names: set[str],
    ctx: SymLinkContext,
) -> bool:
    if not isinstance(node, PyIRCall):
        return False
    if node.args_p or node.args_kw:
        return False
    return _is_auto_func_expr(
        node.func,
        auto_symbol_ids=auto_symbol_ids,
        local_auto_names=local_auto_names,
        ctx=ctx,
    )


def _extract_target_name(ctx: SymLinkContext, t: PyIRNode) -> str | None:
    _ = ctx
    if isinstance(t, PyIRVarUse):
        return t.name

    if isinstance(t, PyIRSymLink):
        if isinstance(t.name, str) and t.name:
            return t.name

    return None


def _resolve_class_symbol_id(
    ctx: SymLinkContext,
    node: PyIRClassDef,
) -> int | None:
    node_uid = ctx.uid(node)
    scope = ctx.node_scope.get(node_uid)
    if scope is None:
        return None

    link = ctx.resolve(scope=scope, name=node.name)
    if link is None:
        return None

    return int(link.target.value)


def _resolve_any_class_id(
    expr: PyIRNode,
    *,
    ctx: SymLinkContext,
    nested_class_ids: dict[tuple[int, str], int],
    current_module: str,
) -> int | None:
    if isinstance(expr, PyIRSymLink) and not expr.is_store:
        return _sid(expr.symbol_id)

    if isinstance(expr, PyIRAttribute):
        parent_id = _resolve_any_class_id(
            expr.value,
            ctx=ctx,
            nested_class_ids=nested_class_ids,
            current_module=current_module,
        )
        if parent_id is not None:
            child_id = nested_class_ids.get((parent_id, expr.attr))
            if child_id is not None:
                return int(child_id)

    parts = _dotted_parts(expr)
    if not parts:
        return None

    if len(parts) == 1:
        sym = ctx.get_exported_symbol(current_module, parts[0])
    else:
        sym = ctx.get_exported_symbol(".".join(parts[:-1]), parts[-1])

    if sym is None:
        return None

    return _sid(sym.id)


def _resolve_enum_class_id(
    expr: PyIRNode,
    *,
    ctx: SymLinkContext,
    store: dict[int, dict[str, _EnumFieldSpec]],
    nested_class_ids: dict[tuple[int, str], int],
    current_module: str,
) -> int | None:
    class_id = _resolve_any_class_id(
        expr,
        ctx=ctx,
        nested_class_ids=nested_class_ids,
        current_module=current_module,
    )
    if class_id is None:
        return None
    return class_id if class_id in store else None


def _extract_target_symbol_ids(target: PyIRNode) -> list[int]:
    out: list[int] = []
    if isinstance(target, PyIRSymLink):
        sid = _sid(target.symbol_id)
        if sid is not None:
            out.append(sid)
    return out


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
        nested_class_ids = _get_nested_class_ids(ctx)

        enum_symbol_kind_by_id: dict[int, str] = {}
        for enum_kind in _ENUM_BASE_NAMES:
            module_ids = collect_symbol_ids_in_modules(
                ctx,
                symbol_name=enum_kind,
                modules=_ENUM_MODULES,
            )
            local_ids = collect_local_imported_symbol_ids(
                ir,
                ctx=ctx,
                current_module=current_module,
                imported_name=enum_kind,
                allowed_modules=_ENUM_MODULES,
            )
            for sid in (*module_ids, *local_ids):
                enum_symbol_kind_by_id[int(sid)] = enum_kind

        auto_symbol_ids = collect_symbol_ids_in_modules(
            ctx,
            symbol_name="auto",
            modules=_ENUM_MODULES,
        )
        auto_symbol_ids |= collect_local_imported_symbol_ids(
            ir,
            ctx=ctx,
            current_module=current_module,
            imported_name="auto",
            allowed_modules=_ENUM_MODULES,
        )

        local_auto_names: set[str] = {"auto"}
        for st in ir.body:
            if not isinstance(st, PyIRImport):
                continue
            if not st.if_from:
                continue
            if ".".join(st.modules or ()) not in _ENUM_MODULES:
                continue
            for src, bound in iter_imported_names(st.names):
                if src == "auto":
                    local_auto_names.add(bound)

        class_defs: list[PyIRClassDef] = [
            n for n in ir.walk() if isinstance(n, PyIRClassDef)
        ]
        class_ids_by_uid: dict[int, int] = {}
        for node in class_defs:
            cls_id = _resolve_class_symbol_id(ctx, node)
            if cls_id is not None:
                class_ids_by_uid[ctx.uid(node)] = cls_id

        for node in class_defs:
            parent_id = class_ids_by_uid.get(ctx.uid(node))
            if parent_id is None:
                continue

            for child in node.body or []:
                if not isinstance(child, PyIRClassDef):
                    continue
                child_id = class_ids_by_uid.get(ctx.uid(child))
                if child_id is None:
                    continue
                nested_class_ids[(int(parent_id), child.name)] = int(child_id)

        for node in class_defs:
            cls_id = class_ids_by_uid.get(ctx.uid(node))
            if cls_id is None:
                continue

            has_gmod_special_enum = False
            for d in node.decorators or []:
                if not _is_gmod_special_enum_decorator(
                    d,
                    core_cd_symids=core_cd_symids,
                    ctx=ctx,
                ):
                    continue
                has_gmod_special_enum = True

                _, _, kw = decorator_call_parts(d)
                fields = kw.get("fields")
                if not isinstance(fields, PyIRDict):
                    CompilerExit.user_error_node(
                        "gmod_special_enum(fields=...) ожидает словарь.",
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

            if has_gmod_special_enum:
                for st in node.body or []:
                    rhs: PyIRNode | None = None
                    if isinstance(st, PyIRAssign) and len(st.targets) == 1:
                        rhs = st.value
                    elif isinstance(st, PyIRAnnotatedAssign):
                        rhs = st.value
                    if rhs is not None and _is_auto_call(
                        rhs,
                        auto_symbol_ids=auto_symbol_ids,
                        local_auto_names=local_auto_names,
                        ctx=ctx,
                    ):
                        CompilerExit.user_error_node(
                            "gmod_special_enum не поддерживает auto(). "
                            "Используйте явные значения полей.",
                            path=ir.path,
                            node=st,
                        )
            enum_kind = _detect_enum_kind(
                node,
                enum_symbol_kind_by_id=enum_symbol_kind_by_id,
            )
            if enum_kind is None:
                continue
            if has_gmod_special_enum:
                continue

            body_map: dict[str, _EnumFieldSpec] = {}
            auto_next_int: int | None = 1

            for st in node.body or []:
                if isinstance(st, PyIRAssign):
                    if len(st.targets) != 1:
                        continue
                    name = _extract_target_name(ctx, st.targets[0])
                    if not name:
                        continue
                    spec = _const_to_spec(st.value)
                    if spec is None and _is_auto_call(
                        st.value,
                        auto_symbol_ids=auto_symbol_ids,
                        local_auto_names=local_auto_names,
                        ctx=ctx,
                    ):
                        if enum_kind == _ENUM_KIND_STR:
                            spec = _EnumFieldSpec("str", name.lower())
                        else:
                            if auto_next_int is None:
                                CompilerExit.user_error_node(
                                    "auto() в Enum/IntEnum требует числовые значения членов.",
                                    path=ir.path,
                                    node=st,
                                )
                            spec = _EnumFieldSpec("int", int(auto_next_int))
                            auto_next_int = int(auto_next_int) + 1

                    if spec:
                        body_map[name] = spec
                        if enum_kind in (_ENUM_KIND_ENUM, _ENUM_KIND_INT):
                            if spec.kind == "int":
                                auto_next_int = int(spec.value) + 1
                            elif spec.kind == "bool":
                                auto_next_int = int(spec.value) + 1
                            elif spec.kind in ("float", "str", "global"):
                                auto_next_int = None

                elif isinstance(st, PyIRAnnotatedAssign):
                    name = st.name
                    spec = _const_to_spec(st.value)
                    if spec is None and _is_auto_call(
                        st.value,
                        auto_symbol_ids=auto_symbol_ids,
                        local_auto_names=local_auto_names,
                        ctx=ctx,
                    ):
                        if enum_kind == _ENUM_KIND_STR:
                            spec = _EnumFieldSpec("str", name.lower())
                        else:
                            if auto_next_int is None:
                                CompilerExit.user_error_node(
                                    "auto() в Enum/IntEnum требует числовые значения членов.",
                                    path=ir.path,
                                    node=st,
                                )
                            spec = _EnumFieldSpec("int", int(auto_next_int))
                            auto_next_int = int(auto_next_int) + 1

                    if spec:
                        body_map[name] = spec
                        if enum_kind in (_ENUM_KIND_ENUM, _ENUM_KIND_INT):
                            if spec.kind == "int":
                                auto_next_int = int(spec.value) + 1
                            elif spec.kind == "bool":
                                auto_next_int = int(spec.value) + 1
                            elif spec.kind in ("float", "str", "global"):
                                auto_next_int = None

            if body_map:
                store.setdefault(cls_id, {}).update(body_map)

        return ir


class FoldGmodSpecialEnumUsesPass:
    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        store = _get_store(ctx)
        if not store or ir.path is None:
            return ir
        nested_class_ids = _get_nested_class_ids(ctx)

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        enum_class_names: set[str] = set()
        for sym in ctx.symbols.values():
            sid = int(sym.id.value)
            if sid in store:
                enum_class_names.add(sym.name)

        enum_typed_var_ids: set[int] = set()
        for sid, atoms in ctx.type_sets_by_symbol_id.items():
            if any(atom in enum_class_names for atom in atoms):
                enum_typed_var_ids.add(int(sid))

        enum_member_var_ids: set[int] = set()
        changed = True
        while changed:
            changed = False
            for st in ir.walk():
                if not isinstance(st, (PyIRAssign, PyIRAnnotatedAssign)):
                    continue

                rhs: PyIRNode
                targets: list[PyIRNode]
                if isinstance(st, PyIRAssign):
                    rhs = st.value
                    targets = st.targets
                else:
                    rhs = st.value
                    targets = []

                rhs_is_enum_member = False
                if isinstance(rhs, PyIRAttribute):
                    cid = _resolve_enum_class_id(
                        rhs.value,
                        ctx=ctx,
                        store=store,
                        nested_class_ids=nested_class_ids,
                        current_module=current_module,
                    )
                    rhs_is_enum_member = bool(
                        cid is not None and rhs.attr in store.get(cid, {})
                    )
                elif isinstance(rhs, PyIRSymLink):
                    rhs_is_enum_member = int(rhs.symbol_id) in enum_member_var_ids

                if not rhs_is_enum_member:
                    continue

                for t in targets:
                    for sid in _extract_target_symbol_ids(t):
                        if sid not in enum_member_var_ids:
                            enum_member_var_ids.add(sid)
                            changed = True

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(node: PyIRNode, store_ctx: bool) -> PyIRNode:
            if isinstance(node, PyIRAttribute):
                if (
                    not store_ctx
                    and node.attr in ("value", "name")
                    and isinstance(node.value, PyIRAttribute)
                ):
                    member = node.value
                    cid = _resolve_enum_class_id(
                        member.value,
                        ctx=ctx,
                        store=store,
                        nested_class_ids=nested_class_ids,
                        current_module=current_module,
                    )
                    if cid is not None:
                        spec = store.get(cid, {}).get(member.attr)
                        if spec is not None:
                            if node.attr == "name":
                                return PyIRConstant(
                                    line=node.line,
                                    offset=node.offset,
                                    value=member.attr,
                                )
                            return _make_replacement(node, spec)

                if (
                    not store_ctx
                    and node.attr == "name"
                    and isinstance(node.value, PyIRSymLink)
                    and int(node.value.symbol_id)
                    in (enum_member_var_ids | enum_typed_var_ids)
                ):
                    CompilerExit.user_error_node(
                        "Доступ к .name разрешён только как EnumClass.Member.name. "
                        "Для переменной-члена enum это запрещено.",
                        path=ir.path,
                        node=node,
                    )

                if (
                    not store_ctx
                    and node.attr == "value"
                    and isinstance(node.value, PyIRSymLink)
                    and int(node.value.symbol_id)
                    in (enum_member_var_ids | enum_typed_var_ids)
                ):
                    return node.value

                node.value = rw_expr(node.value, store_ctx=False)

                if not store_ctx:
                    cid = _resolve_enum_class_id(
                        node.value,
                        ctx=ctx,
                        store=store,
                        nested_class_ids=nested_class_ids,
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

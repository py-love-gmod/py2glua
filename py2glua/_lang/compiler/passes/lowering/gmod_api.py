from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, cast

from ....._cli import CompilerExit, logger
from ....compiler.passes.analysis.symlinks import PyIRSymLink, SymLinkContext
from ....py.ir_dataclass import (
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
    PyIRFunctionDef,
    PyIRList,
    PyIRNode,
    PyIRTuple,
    PyIRVarUse,
)
from ..common import (
    collect_core_compiler_directive_local_symbol_ids,
    collect_core_compiler_directive_symbol_ids,
    decorator_call_parts,
    is_expr_on_symbol_ids,
    resolve_symbol_ids_by_attr_chain,
)
from ..rewrite_utils import rewrite_expr_with_store_default, rewrite_stmt_block


@dataclass(frozen=True)
class GmodApiDecl:
    lua_name: str
    method: bool
    decl_file: object
    decl_line: int | None
    py_params: tuple[str, ...] = ()
    py_defaults: Dict[str, PyIRNode] = field(default_factory=dict)
    py_vararg: str | None = None
    py_kwarg: str | None = None
    base_args: tuple[PyIRNode, ...] = ()


@dataclass
class GmodApiClassInfo:
    symbol_id: int
    name: str
    bases: tuple[int, ...] = ()
    methods: Dict[str, GmodApiDecl] = field(default_factory=dict)


def _warn_once(ctx: SymLinkContext, key: str, msg: str) -> None:
    warned = ctx._gmod_api_warned
    if key in warned:
        return

    warned.add(key)
    logger.warning(msg)


def _ensure_storage(
    ctx: SymLinkContext,
) -> tuple[
    dict[int, GmodApiDecl],
    dict[int, GmodApiClassInfo],
    dict[tuple[int, str], GmodApiDecl],
    dict[str, int],
    bool,
]:
    return (
        cast(dict[int, GmodApiDecl], ctx.gmod_api_funcs),
        cast(dict[int, GmodApiClassInfo], ctx.gmod_api_classes),
        cast(dict[tuple[int, str], GmodApiDecl], ctx.gmod_api_resolved_methods),
        ctx.gmod_api_name_to_class_id,
        ctx._gmod_api_registry_ready,
    )


def _unwrap_gmod_api_decorator(
    dec: PyIRDecorator,
) -> tuple[PyIRAttribute, list[PyIRNode], dict[str, PyIRNode]] | None:
    """
    Two formats:
      A) PyIRDecorator(exper=Attr(CD, gmod_api), args_p/args_kw)
      B) PyIRDecorator(exper=Call(func=Attr(CD, gmod_api), args_p/args_kw), args inside call)
    """
    fn_expr, args_p, args_kw = decorator_call_parts(dec)
    if isinstance(fn_expr, PyIRAttribute):
        return fn_expr, args_p, args_kw
    return None


def _parse_gmod_api(
    ir: PyIRFile,
    core_cd_symids: set[int],
    ctx: SymLinkContext,
    dec: PyIRDecorator,
    ref: PyIRNode,
) -> GmodApiDecl | None:
    u = _unwrap_gmod_api_decorator(dec)
    if u is None:
        return None

    func_attr, args_p, args_kw = u

    if func_attr.attr != "gmod_api":
        return None

    if not is_expr_on_symbol_ids(
        func_attr.value,
        symbol_ids=core_cd_symids,
        ctx=ctx,
    ):
        return None

    # Signature: gmod_api(name, realm, method=False)
    allowed_kw = {"name", "realm", "method", "base_args"}
    for k in args_kw.keys():
        if k not in allowed_kw:
            CompilerExit.user_error_node(
                f"Неизвестный keyword-аргумент '{k}' в gmod_api.",
                ir.path,
                ref,
            )

    if len(args_p) > 3:
        CompilerExit.user_error_node(
            "gmod_api ожидает не более 3 позиционных аргументов (name, realm, method).",
            ir.path,
            ref,
        )

    def pick_arg(idx: int, key: str) -> PyIRNode | None:
        if idx < len(args_p):
            if key in args_kw:
                CompilerExit.user_error_node(
                    f"Аргумент '{key}' передан дважды (positional + keyword) в gmod_api.",
                    ir.path,
                    ref,
                )
            return args_p[idx]
        return args_kw.get(key)

    name_node = pick_arg(0, "name")
    realm_node = pick_arg(1, "realm")
    method_node = pick_arg(2, "method")
    base_args_node = args_kw.get("base_args")

    if name_node is None:
        CompilerExit.user_error_node(
            "gmod_api ожидает аргумент name.",
            ir.path,
            ref,
        )

    if not isinstance(name_node, PyIRConstant) or not isinstance(name_node.value, str):
        CompilerExit.user_error_node(
            "Декоратор gmod_api ожидает строковый литерал в аргументе name.",
            ir.path,
            ref,
        )

    if realm_node is None:
        CompilerExit.user_error_node(
            "gmod_api ожидает аргумент realm.",
            ir.path,
            ref,
        )

    lua_name = name_node.value

    method_val = False
    if method_node is not None:
        if not isinstance(method_node, PyIRConstant) or not isinstance(
            method_node.value, bool
        ):
            CompilerExit.user_error_node(
                "Параметр method в gmod_api должен быть булевым литералом (True/False).",
                ir.path,
                ref,
            )

        method_val = bool(method_node.value)

    py_params: tuple[str, ...] = ()
    py_defaults: Dict[str, PyIRNode] = {}
    py_vararg: str | None = None
    py_kwarg: str | None = None
    base_args: tuple[PyIRNode, ...] = ()
    if isinstance(ref, PyIRFunctionDef):
        py_params = tuple(ref.signature.keys())
        py_defaults = {
            pname: default
            for pname, (_ann, default) in ref.signature.items()
            if default is not None
        }
        py_vararg = ref.vararg
        py_kwarg = ref.kwarg

    if base_args_node is not None:
        if isinstance(base_args_node, (PyIRList, PyIRTuple)):
            base_args = tuple(base_args_node.elements)
        else:
            CompilerExit.user_error_node(
                "Параметр base_args в gmod_api должен быть list/tuple литералом.",
                ir.path,
                ref,
            )

    return GmodApiDecl(
        lua_name=lua_name,
        method=method_val,
        decl_file=ir.path,
        decl_line=ref.line,
        py_params=py_params,
        py_defaults=py_defaults,
        py_vararg=py_vararg,
        py_kwarg=py_kwarg,
        base_args=base_args,
    )


def _make_emit_call(ref: PyIRNode, lua_name: str, args: list[PyIRNode]) -> PyIREmitExpr:
    return PyIREmitExpr(
        line=ref.line,
        offset=ref.offset,
        kind=PyIREmitKind.CALL,
        name=lua_name,
        args_p=args,
        args_kw={},
    )


def _make_emit_method_call(
    ref: PyIRNode,
    lua_method_name: str,
    base_obj: PyIRNode,
    args: list[PyIRNode],
) -> PyIREmitExpr:
    return PyIREmitExpr(
        line=ref.line,
        offset=ref.offset,
        kind=PyIREmitKind.METHOD_CALL,
        name=lua_method_name,
        args_p=[base_obj, *args],
        args_kw={},
    )


class CollectGmodApiDeclsPass:
    """
    Collects gmod_api declarations into ctx.

    Fills:
      - ctx.gmod_api_funcs[symbol_id] = decl   (top-level functions)
      - ctx.gmod_api_classes[class_id] = info (bases + declared methods)
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        funcs, classes, _resolved, _name_map, _ready = _ensure_storage(ctx)
        core_cd_symids = set(collect_core_compiler_directive_symbol_ids(ctx))

        file_uid = ctx.uid(ir)
        file_scope = ctx.file_scope.get(file_uid)
        if file_scope is None:
            CompilerExit.internal_error(
                "Не удалось получить file_scope для файла при сборе gmod_api.\n"
                f"Файл: {ir.path}"
            )

        def resolve_symbol_id_in_file(name: str) -> int | None:
            link = ctx.resolve(scope=file_scope, name=name)
            return link.target.value if link is not None else None

        if ir.path is not None:
            current_module = ctx.module_name_by_path.get(ir.path)
            if current_module:
                core_cd_symids |= collect_core_compiler_directive_local_symbol_ids(
                    ir,
                    ctx=ctx,
                    current_module=current_module,
                )

        for st in ir.body:
            if isinstance(st, PyIRFunctionDef):
                decl: GmodApiDecl | None = None
                for d in st.decorators:
                    decl = _parse_gmod_api(ir, core_cd_symids, ctx, d, st)
                    if decl is not None:
                        break
                if decl is None:
                    continue

                sid = resolve_symbol_id_in_file(st.name)
                if sid is None:
                    CompilerExit.internal_error(
                        "Не удалось определить symbol_id для функции с gmod_api.\n"
                        f"Файл: {ir.path}\n"
                        f"Функция: {st.name}"
                    )

                funcs[sid] = decl
                continue

            if isinstance(st, PyIRClassDef):
                cid = resolve_symbol_id_in_file(st.name)
                if cid is None:
                    CompilerExit.internal_error(
                        "Не удалось определить symbol_id для класса.\n"
                        f"Файл: {ir.path}\n"
                        f"Класс: {st.name}"
                    )

                info = classes.get(cid)
                if info is None:
                    info = GmodApiClassInfo(symbol_id=cid, name=st.name)
                    classes[cid] = info

                base_ids: list[int] = []
                for b in st.bases:
                    if isinstance(b, PyIRSymLink):
                        base_ids.append(b.symbol_id)

                    elif isinstance(b, PyIRVarUse):
                        bid = resolve_symbol_id_in_file(b.name)
                        if bid is not None:
                            base_ids.append(bid)

                info.bases = tuple(base_ids)

                for m in st.body:
                    if not isinstance(m, PyIRFunctionDef):
                        continue

                    mdecl: GmodApiDecl | None = None
                    for d in m.decorators:
                        mdecl = _parse_gmod_api(ir, core_cd_symids, ctx, d, m)
                        if mdecl is not None:
                            break

                    if mdecl is None:
                        continue

                    info.methods[m.name] = mdecl

        ctx._gmod_api_registry_ready = False
        return ir


class FinalizeGmodApiRegistryPass:
    """
    Builds:
      - ctx.gmod_api_name_to_class_id
      - ctx.gmod_api_resolved_methods[(class_id, method_name)] = decl
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        _funcs, classes, resolved, name_map, ready = _ensure_storage(ctx)
        _ = ir

        if ready:
            return ir

        name_map.clear()
        for cid, info in classes.items():
            name_map[info.name] = cid

        resolved.clear()

        visiting: set[int] = set()
        memo: Dict[int, Dict[str, GmodApiDecl]] = {}

        def resolve_all_methods(cid: int) -> Dict[str, GmodApiDecl]:
            m = memo.get(cid)
            if m is not None:
                return m

            if cid in visiting:
                CompilerExit.internal_error(
                    "Обнаружен цикл наследования в gmod_api классах (циклические bases)."
                )

            visiting.add(cid)

            info = classes.get(cid)
            if info is None:
                visiting.remove(cid)
                memo[cid] = {}
                return {}

            merged: Dict[str, GmodApiDecl] = {}

            for bid in info.bases:
                bm = resolve_all_methods(bid)
                for k, v in bm.items():
                    if k not in merged:
                        merged[k] = v

            for k, v in info.methods.items():
                merged[k] = v

            visiting.remove(cid)
            memo[cid] = merged
            return merged

        for cid in list(classes.keys()):
            mm = resolve_all_methods(cid)
            for mname, decl in mm.items():
                resolved[(cid, mname)] = decl

        ctx._gmod_api_registry_ready = True
        return ir


class RewriteGmodApiCallsPass:
    """
    Rewrites gmod_api calls.

    IMPORTANT:
      - This pass now traverses the whole IR (functions, ifs, loops, with, classes).
      - Receiver types are taken from:
          ctx.type_env_by_uid[uid(call)][symbol_id] (flow snapshot)
        fallback:
          ctx.type_sets_by_symbol_id[symbol_id]

    Type atoms (from TypeFlowPass):
      - "None", "Nil", "Value"
    """

    ATOM_NONE = "None"
    ATOM_NIL = "Nil"
    ATOM_VALUE = "Value"

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        funcs, classes, resolved, name_map, ready = _ensure_storage(ctx)
        if not ready:
            FinalizeGmodApiRegistryPass.run(ir, ctx)
            funcs, classes, resolved, name_map, _ready = _ensure_storage(ctx)

        env_by_uid = ctx.type_env_by_uid
        base_types = ctx.type_sets_by_symbol_id

        def types_at_call(call: PyIRCall, sym_id: int) -> frozenset[str] | None:
            snap = env_by_uid.get(ctx.uid(call))
            if isinstance(snap, dict):
                t = snap.get(sym_id)
                if isinstance(t, frozenset):
                    return t

            t2 = base_types.get(sym_id)
            if isinstance(t2, frozenset):
                return t2

            return None

        def candidate_class_ids(tset: frozenset[str]) -> tuple[list[int], bool]:
            had_nullish = (RewriteGmodApiCallsPass.ATOM_NONE in tset) or (
                RewriteGmodApiCallsPass.ATOM_NIL in tset
            )

            ids: list[int] = []
            for n in tset:
                if n in (
                    RewriteGmodApiCallsPass.ATOM_NONE,
                    RewriteGmodApiCallsPass.ATOM_NIL,
                    RewriteGmodApiCallsPass.ATOM_VALUE,
                ):
                    continue
                cid = name_map.get(n)
                if cid is not None:
                    ids.append(cid)

            out: list[int] = []
            seen: set[int] = set()
            for i in ids:
                if i in seen:
                    continue

                seen.add(i)
                out.append(i)

            return out, had_nullish

        def pick_single_decl(
            decls: list[GmodApiDecl],
            *,
            mname: str,
            node: PyIRNode,
        ) -> GmodApiDecl:
            first = decls[0]
            for d in decls[1:]:
                if (
                    d.lua_name != first.lua_name
                    or d.method != first.method
                    or d.py_params != first.py_params
                    or d.py_vararg != first.py_vararg
                    or d.py_kwarg != first.py_kwarg
                    or tuple(d.py_defaults.keys()) != tuple(first.py_defaults.keys())
                    or len(d.base_args) != len(first.base_args)
                ):
                    CompilerExit.user_error_node(
                        "Не удалось однозначно определить, какой gmod_api-вызов использовать.\n"
                        f"Имя метода: {mname}\n"
                        f"Подходящие варианты: {', '.join(dd.lua_name for dd in decls)}\n"
                        "Уточни тип объекта (receiver), чтобы компилятор выбрал один вариант.",
                        ir.path,
                        node,
                    )
            return first

        def pick_fallback_object_method_decl(
            *,
            mname: str,
            node: PyIRNode,
        ) -> GmodApiDecl | None:
            by_name = [d for (cid, mn), d in resolved.items() if mn == mname]
            if not by_name:
                return None

            method_decls = [d for d in by_name if d.method]
            func_decls = [d for d in by_name if not d.method]

            if not method_decls:
                return None

            if func_decls:
                CompilerExit.user_error_node(
                    f"Нельзя безопасно переписать вызов '{mname}'.\n"
                    "В gmod_api это имя объявлено и как метод (`method=True`), "
                    "и как обычная функция (`method=False`).\n"
                    "Добавь или уточни тип объекта, у которого вызывается метод.",
                    ir.path,
                    node,
                )

            decl = pick_single_decl(method_decls, mname=mname, node=node)
            _warn_once(
                ctx,
                f"gmod_api_heuristic_method_{ctx.uid(node)}",
                f"WARN: Вызов '{mname}' переписан как gmod_api-метод по эвристике.\n"
                "Тип объекта не удалось вывести точно, поэтому выбран `obj:method(...)`.\n"
                f"Файл: {ir.path}\nLINE|OFFSET: {node.line}|{node.offset}",
            )
            return decl

        def normalize_call_args_for_decl(
            call: PyIRCall,
            decl: GmodApiDecl,
            *,
            implicit_first_param: bool,
            display_name: str,
        ) -> PyIRCall:
            params = list(decl.py_params)
            if implicit_first_param:
                if not params:
                    CompilerExit.internal_error(
                        "gmod_api: метод объявлен с method=True, но сигнатура пуста.\n"
                        f"Вызов: {display_name}\n"
                        f"Файл объявления: {decl.decl_file}:{decl.decl_line}"
                    )
                params = params[1:]

            defaults = decl.py_defaults
            provided: Dict[str, PyIRNode] = {}
            extra_pos: list[PyIRNode] = []

            for i, arg in enumerate(call.args_p):
                if i < len(params):
                    provided[params[i]] = arg
                else:
                    extra_pos.append(arg)

            extra_kw: Dict[str, PyIRNode] = {}
            for key, value in call.args_kw.items():
                if key in params:
                    if key in provided:
                        CompilerExit.user_error_node(
                            f"Аргумент '{key}' передан дважды при вызове '{display_name}' "
                            "(позиционно и по имени).",
                            ir.path,
                            call,
                        )
                    provided[key] = value
                else:
                    extra_kw[key] = value

            if extra_pos and decl.py_vararg is None:
                CompilerExit.user_error_node(
                    f"Слишком много позиционных аргументов при вызове '{display_name}'.",
                    ir.path,
                    call,
                )

            if extra_kw and decl.py_kwarg is None:
                bad_names = ", ".join(sorted(extra_kw.keys()))
                CompilerExit.user_error_node(
                    f"Неизвестные keyword-аргументы при вызове '{display_name}': {bad_names}.",
                    ir.path,
                    call,
                )

            new_args_p: list[PyIRNode] = []
            if decl.base_args:
                new_args_p.extend(deepcopy(arg) for arg in decl.base_args)

            for param_name in params:
                if param_name in provided:
                    new_args_p.append(provided[param_name])
                    continue

                default_value = defaults.get(param_name)
                if default_value is None:
                    CompilerExit.user_error_node(
                        f"Не хватает обязательного аргумента '{param_name}' "
                        f"при вызове '{display_name}'.",
                        ir.path,
                        call,
                    )

                new_args_p.append(deepcopy(default_value))

            if decl.py_vararg is not None:
                new_args_p.append(
                    PyIRList(
                        line=call.line,
                        offset=call.offset,
                        elements=extra_pos,
                    )
                )

            if decl.py_kwarg is not None:
                kw_items: list[PyIRDictItem] = [
                    PyIRDictItem(
                        line=value.line,
                        offset=value.offset,
                        key=PyIRConstant(
                            line=value.line,
                            offset=value.offset,
                            value=key,
                        ),
                        value=value,
                    )
                    for key, value in extra_kw.items()
                ]
                new_args_p.append(
                    PyIRDict(
                        line=call.line,
                        offset=call.offset,
                        items=kw_items,
                    )
                )

            return PyIRCall(
                line=call.line,
                offset=call.offset,
                func=call.func,
                args_p=new_args_p,
                args_kw={},
            )

        def rw_block(stmts: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(stmts, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(node, PyIRCall):
                node.func = rw_expr(node.func, False)
                node.args_p = [rw_expr(a, False) for a in node.args_p]
                node.args_kw = {k: rw_expr(v, False) for k, v in node.args_kw.items()}

                if isinstance(node.func, PyIRSymLink):
                    decl = funcs.get(node.func.symbol_id)
                    if decl is not None:
                        node = normalize_call_args_for_decl(
                            node,
                            decl,
                            implicit_first_param=False,
                            display_name=decl.lua_name,
                        )
                        return _make_emit_call(node, decl.lua_name, node.args_p)
                else:
                    fn_symids = resolve_symbol_ids_by_attr_chain(ctx, node.func)
                    fn_decls = [funcs[sid] for sid in fn_symids if sid in funcs]
                    if fn_decls:
                        decl = pick_single_decl(
                            fn_decls,
                            mname=getattr(node.func, "attr", "<call>"),
                            node=node,
                        )
                        node = normalize_call_args_for_decl(
                            node,
                            decl,
                            implicit_first_param=False,
                            display_name=decl.lua_name,
                        )
                        return _make_emit_call(node, decl.lua_name, node.args_p)

                # Constructor form: Class(...)
                # Supports normalized imports like: utils.Angle(...)
                ctor_cls_symids = [
                    sid
                    for sid in resolve_symbol_ids_by_attr_chain(ctx, node.func)
                    if sid in classes
                ]
                if ctor_cls_symids:
                    ctor_decls: list[GmodApiDecl] = []
                    for cid in ctor_cls_symids:
                        info = classes.get(cid)
                        if info is None:
                            continue

                        if (
                            isinstance(node.func, PyIRAttribute)
                            and node.func.attr != info.name
                        ):
                            continue

                        d = resolved.get((cid, "__init__"))
                        if d is not None:
                            ctor_decls.append(d)

                    if ctor_decls:
                        ctor_decl = pick_single_decl(
                            ctor_decls,
                            mname="__init__",
                            node=node,
                        )

                        if ctor_decl.method:
                            CompilerExit.user_error_node(
                                "Конструктор gmod_api (`__init__`) не может быть объявлен как "
                                "метод (`method=True`).\n"
                                "Для конструктора используй `method=False`.",
                                ir.path,
                                node,
                            )

                        node = normalize_call_args_for_decl(
                            node,
                            ctor_decl,
                            implicit_first_param=False,
                            display_name=ctor_decl.lua_name,
                        )
                        return _make_emit_call(node, ctor_decl.lua_name, node.args_p)

                if isinstance(node.func, PyIRAttribute):
                    base = node.func.value
                    mname = node.func.attr

                    cls_symids = [
                        sid
                        for sid in resolve_symbol_ids_by_attr_chain(ctx, base)
                        if sid in classes
                    ]
                    if cls_symids:
                        class_decls = [
                            resolved[(cid, mname)]
                            for cid in cls_symids
                            if (cid, mname) in resolved
                        ]
                        if class_decls:
                            first = pick_single_decl(
                                class_decls,
                                mname=mname,
                                node=node,
                            )
                            if first.method:
                                CompilerExit.user_error_node(
                                    f"'{mname}' объявлен в gmod_api как метод (`method=True`), "
                                    "но вызывается как `Class.method(...)`.\n"
                                    f"Используй вызов у объекта: `obj.{mname}(...)` "
                                    "(компилятор сам превратит его в `obj:{mname}(...)`).",
                                    ir.path,
                                    node,
                                )
                            node = normalize_call_args_for_decl(
                                node,
                                first,
                                implicit_first_param=False,
                                display_name=first.lua_name,
                            )
                            return _make_emit_call(node, first.lua_name, node.args_p)

                    if not isinstance(base, PyIRSymLink):
                        fallback_decl = pick_fallback_object_method_decl(
                            mname=mname,
                            node=node,
                        )
                        if fallback_decl is None:
                            return node

                        node = normalize_call_args_for_decl(
                            node,
                            fallback_decl,
                            implicit_first_param=True,
                            display_name=fallback_decl.lua_name,
                        )
                        lua_method_name = fallback_decl.lua_name.rsplit(".", 1)[-1]
                        return _make_emit_method_call(
                            node,
                            lua_method_name,
                            base,
                            node.args_p,
                        )

                    tset = types_at_call(node, base.symbol_id)
                    if tset is None:
                        fallback_decl = pick_fallback_object_method_decl(
                            mname=mname,
                            node=node,
                        )
                        if fallback_decl is None:
                            return node

                        node = normalize_call_args_for_decl(
                            node,
                            fallback_decl,
                            implicit_first_param=True,
                            display_name=fallback_decl.lua_name,
                        )
                        lua_method_name = fallback_decl.lua_name.rsplit(".", 1)[-1]
                        return _make_emit_method_call(
                            node,
                            lua_method_name,
                            base,
                            node.args_p,
                        )
                    cls_ids, had_nullish = candidate_class_ids(tset)

                    if had_nullish:
                        _warn_once(
                            ctx,
                            f"nullish_call_{ir.path}_{node.line}_{node.offset}",
                            f"WARN: Вызов '{mname}' может выполняться на `None` или `nil`.\n"
                            "Проверь, что перед вызовом есть защита от nil/None.\n"
                            f"Файл: {ir.path}\nLINE|OFFSET: {node.line}|{node.offset}",
                        )

                    if not cls_ids:
                        fallback_decl = pick_fallback_object_method_decl(
                            mname=mname,
                            node=node,
                        )
                        if fallback_decl is None:
                            return node

                        node = normalize_call_args_for_decl(
                            node,
                            fallback_decl,
                            implicit_first_param=True,
                            display_name=fallback_decl.lua_name,
                        )
                        lua_method_name = fallback_decl.lua_name.rsplit(".", 1)[-1]
                        return _make_emit_method_call(
                            node,
                            lua_method_name,
                            base,
                            node.args_p,
                        )

                    object_decls: list[GmodApiDecl] = []
                    for cid in cls_ids:
                        d = resolved.get((cid, mname))
                        if d is None:
                            info = classes.get(cid)
                            cname = info.name if info is not None else f"<sym {cid}>"
                            CompilerExit.user_error_node(
                                f"Метод '{mname}' не найден в gmod_api для типа '{cname}'.\n"
                                "Проверь имя метода и тип объекта.",
                                ir.path,
                                node,
                            )

                        object_decls.append(d)

                    first = pick_single_decl(object_decls, mname=mname, node=node)

                    if not first.method:
                        CompilerExit.user_error_node(
                            f"'{mname}' объявлен в gmod_api как обычная функция "
                            "(`method=False`), но вызывается как метод объекта.\n"
                            f"Используй вызов функции, а не `obj.{mname}(...)`.",
                            ir.path,
                            node,
                        )

                    node = normalize_call_args_for_decl(
                        node,
                        first,
                        implicit_first_param=True,
                        display_name=first.lua_name,
                    )
                    lua_method_name = first.lua_name.rsplit(".", 1)[-1]
                    return _make_emit_method_call(
                        node,
                        lua_method_name,
                        base,
                        node.args_p,
                    )

                return node

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

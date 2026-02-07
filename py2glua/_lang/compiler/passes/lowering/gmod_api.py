from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ....._cli import CompilerExit, logger
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
from ...compiler_ir import PyIRFunctionExpr


@dataclass(frozen=True)
class GmodApiDecl:
    lua_name: str
    method: bool
    decl_file: object
    decl_line: int | None


@dataclass
class GmodApiClassInfo:
    symbol_id: int
    name: str
    bases: tuple[int, ...] = ()
    methods: Dict[str, GmodApiDecl] = field(default_factory=dict)


def _warn_once(ctx: SymLinkContext, key: str, msg: str) -> None:
    warned = getattr(ctx, "_gmod_api_warned", None)
    if not isinstance(warned, set):
        warned = set()
        setattr(ctx, "_gmod_api_warned", warned)

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
    funcs = getattr(ctx, "gmod_api_funcs", None)
    if not isinstance(funcs, dict):
        funcs = {}

    classes = getattr(ctx, "gmod_api_classes", None)
    if not isinstance(classes, dict):
        classes = {}

    resolved = getattr(ctx, "gmod_api_resolved_methods", None)
    if not isinstance(resolved, dict):
        resolved = {}

    name_map = getattr(ctx, "gmod_api_name_to_class_id", None)
    if not isinstance(name_map, dict):
        name_map = {}

    ready = getattr(ctx, "_gmod_api_registry_ready", False)
    if not isinstance(ready, bool):
        ready = False

    setattr(ctx, "gmod_api_funcs", funcs)
    setattr(ctx, "gmod_api_classes", classes)
    setattr(ctx, "gmod_api_resolved_methods", resolved)
    setattr(ctx, "gmod_api_name_to_class_id", name_map)
    setattr(ctx, "_gmod_api_registry_ready", ready)

    return funcs, classes, resolved, name_map, ready


def _collect_compiler_directive_local_names(ir: PyIRFile) -> set[str]:
    """
    Supports:
      from py2glua.glua... import CompilerDirective
      from py2glua.glua... import CompilerDirective as CD
    """
    names: set[str] = {"CompilerDirective"}

    for n in ir.body:
        if not isinstance(n, PyIRImport):
            continue

        if not n.if_from:
            continue

        mod = ".".join(n.modules)
        if "py2glua.glua" not in mod:
            continue

        for item in n.names:
            if isinstance(item, tuple):
                src, alias = item
                if src == "CompilerDirective":
                    names.add(alias)

            else:
                if item == "CompilerDirective":
                    names.add(item)

    return names


def _unwrap_gmod_api_decorator(
    dec: PyIRDecorator,
) -> tuple[PyIRAttribute, list[PyIRNode], dict[str, PyIRNode]] | None:
    """
    Two formats:
      A) PyIRDecorator(exper=Attr(CD, gmod_api), args_p/args_kw)
      B) PyIRDecorator(exper=Call(func=Attr(CD, gmod_api), args_p/args_kw), args inside call)
    """
    exp = dec.exper

    if isinstance(exp, PyIRAttribute):
        return exp, dec.args_p, dec.args_kw

    if isinstance(exp, PyIRCall) and isinstance(exp.func, PyIRAttribute):
        return exp.func, exp.args_p, exp.args_kw

    return None


def _is_cd_ref(n: PyIRNode, cd_names: set[str]) -> bool:
    if isinstance(n, PyIRVarUse):
        return n.name in cd_names

    if isinstance(n, PyIRSymLink):
        return n.name in cd_names

    return False


def _parse_gmod_api(
    ir: PyIRFile,
    cd_names: set[str],
    dec: PyIRDecorator,
    ref: PyIRNode,
) -> GmodApiDecl | None:
    u = _unwrap_gmod_api_decorator(dec)
    if u is None:
        return None

    func_attr, args_p, args_kw = u

    if func_attr.attr != "gmod_api":
        return None

    if not _is_cd_ref(func_attr.value, cd_names):
        return None

    name_node: PyIRNode | None = None
    if args_p:
        name_node = args_p[0]
    elif "name" in args_kw:
        name_node = args_kw["name"]

    if not isinstance(name_node, PyIRConstant) or not isinstance(name_node.value, str):
        CompilerExit.user_error_node(
            "Декоратор gmod_api ожидает строковую константу в аргументе name.",
            ir.path,
            ref,
        )

    lua_name = name_node.value

    method_val = False
    method_node: PyIRNode | None = None
    if "method" in args_kw:
        method_node = args_kw["method"]

    elif len(args_p) >= 2:
        method_node = args_p[1]

    if method_node is not None:
        if not isinstance(method_node, PyIRConstant) or not isinstance(
            method_node.value, bool
        ):
            CompilerExit.user_error_node(
                "Параметр method в gmod_api должен быть булевой константой (True/False).",
                ir.path,
                ref,
            )

        method_val = bool(method_node.value)

    return GmodApiDecl(
        lua_name=lua_name,
        method=method_val,
        decl_file=ir.path,
        decl_line=ref.line,
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

        cd_names = _collect_compiler_directive_local_names(ir)

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

        for st in ir.body:
            if isinstance(st, PyIRFunctionDef):
                decl: GmodApiDecl | None = None
                for d in st.decorators:
                    decl = _parse_gmod_api(ir, cd_names, d, st)
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
                        mdecl = _parse_gmod_api(ir, cd_names, d, m)
                        if mdecl is not None:
                            break

                    if mdecl is None:
                        continue

                    info.methods[m.name] = mdecl

        setattr(ctx, "_gmod_api_registry_ready", False)
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

        setattr(ctx, "_gmod_api_registry_ready", True)
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

        env_by_uid = getattr(ctx, "type_env_by_uid", None)
        if not isinstance(env_by_uid, dict):
            env_by_uid = {}

        base_types = getattr(ctx, "type_sets_by_symbol_id", None)
        if not isinstance(base_types, dict):
            base_types = {}

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

        def rw_expr(node: PyIRNode) -> PyIRNode:
            if isinstance(node, PyIRCall):
                node.func = rw_expr(node.func)
                node.args_p = [rw_expr(a) for a in node.args_p]
                node.args_kw = {k: rw_expr(v) for k, v in node.args_kw.items()}

                if node.args_kw:
                    CompilerExit.user_error_node(
                        "Для вызовов gmod_api ключевые аргументы должны быть предварительно нормализованы.",
                        ir.path,
                        node,
                    )

                if isinstance(node.func, PyIRSymLink):
                    decl = funcs.get(node.func.symbol_id)
                    if decl is not None:
                        return _make_emit_call(node, decl.lua_name, node.args_p)

                if isinstance(node.func, PyIRAttribute):
                    base = node.func.value
                    mname = node.func.attr

                    if not isinstance(base, PyIRSymLink):
                        return node

                    tset = types_at_call(node, base.symbol_id)
                    if tset is None:
                        return node

                    cls_ids, had_nullish = candidate_class_ids(tset)

                    if had_nullish:
                        _warn_once(
                            ctx,
                            f"nullish_call_{ir.path}_{node.line}_{node.offset}",
                            f"WARN: Вызов '{mname}' может выполняться на None|Nil\n"
                            f"Файл: {ir.path}\nLINE|OFFSET: {node.line}|{node.offset}",
                        )

                    if not cls_ids:
                        return node

                    decls: list[GmodApiDecl] = []
                    for cid in cls_ids:
                        d = resolved.get((cid, mname))
                        if d is None:
                            cname = (
                                classes.get(cid).name  # type: ignore
                                if cid in classes
                                else f"<sym {cid}>"
                            )
                            CompilerExit.user_error_node(
                                f"Метод '{mname}' не найден в gmod_api для типа '{cname}'.",
                                ir.path,
                                node,
                            )

                        decls.append(d)

                    first = decls[0]
                    for d in decls[1:]:
                        if d.lua_name != first.lua_name or d.method != first.method:
                            CompilerExit.user_error_node(
                                "Невозможно однозначно разрешить вызов gmod_api метода.\n"
                                f"Метод: {mname}\n"
                                f"Варианты: {', '.join(dd.lua_name for dd in decls)}",
                                ir.path,
                                node,
                            )

                    if not first.method:
                        CompilerExit.user_error_node(
                            f"'{mname}' помечен gmod_api как функция (method=False), но вызывается как obj.{mname}(...).",
                            ir.path,
                            node,
                        )

                    return _make_emit_call(node, first.lua_name, [base, *node.args_p])

                return node

            if isinstance(node, PyIRAttribute):
                node.value = rw_expr(node.value)
                return node

            if isinstance(node, PyIRUnaryOP):
                node.value = rw_expr(node.value)
                return node

            if isinstance(node, PyIRBinOP):
                node.left = rw_expr(node.left)
                node.right = rw_expr(node.right)
                return node

            if isinstance(node, PyIRSubscript):
                node.value = rw_expr(node.value)
                node.index = rw_expr(node.index)
                return node

            if isinstance(node, PyIRList):
                node.elements = [rw_expr(e) for e in node.elements]
                return node

            if isinstance(node, PyIRTuple):
                node.elements = [rw_expr(e) for e in node.elements]
                return node

            if isinstance(node, PyIRSet):
                node.elements = [rw_expr(e) for e in node.elements]
                return node

            if isinstance(node, PyIRDictItem):
                node.key = rw_expr(node.key)
                node.value = rw_expr(node.value)
                return node

            if isinstance(node, PyIRDict):
                node.items = [rw_expr(i) for i in node.items]  # type: ignore[list-item]
                return node

            if isinstance(node, PyIRDecorator):
                node.exper = rw_expr(node.exper)
                node.args_p = [rw_expr(a) for a in node.args_p]
                node.args_kw = {k: rw_expr(v) for k, v in node.args_kw.items()}
                return node

            if isinstance(node, PyIREmitExpr):
                node.args_p = [rw_expr(a) for a in node.args_p]
                node.args_kw = {k: rw_expr(v) for k, v in node.args_kw.items()}
                return node

            return node

        def rw_stmt_list(stmts: list[PyIRNode]) -> list[PyIRNode]:
            out: list[PyIRNode] = []
            for s in stmts:
                out.extend(rw_stmt(s))
            return out

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            if isinstance(st, PyIRIf):
                st.test = rw_expr(st.test)
                st.body = rw_stmt_list(st.body)
                st.orelse = rw_stmt_list(st.orelse)
                return [st]

            if isinstance(st, PyIRWhile):
                st.test = rw_expr(st.test)
                st.body = rw_stmt_list(st.body)
                return [st]

            if isinstance(st, PyIRFor):
                st.target = rw_expr(st.target)
                st.iter = rw_expr(st.iter)
                st.body = rw_stmt_list(st.body)
                return [st]

            if isinstance(st, PyIRWithItem):
                st.context_expr = rw_expr(st.context_expr)
                if st.optional_vars is not None:
                    st.optional_vars = rw_expr(st.optional_vars)
                return [st]

            if isinstance(st, PyIRWith):
                st.items = [rw_expr(i) for i in st.items]  # type: ignore[list-item]
                st.body = rw_stmt_list(st.body)
                return [st]

            if isinstance(st, PyIRAssign):
                st.targets = [rw_expr(t) for t in st.targets]
                st.value = rw_expr(st.value)
                return [st]

            if isinstance(st, PyIRAugAssign):
                st.target = rw_expr(st.target)
                st.value = rw_expr(st.value)
                return [st]

            if isinstance(st, PyIRReturn):
                st.value = rw_expr(st.value)  # pyright: ignore[reportArgumentType]
                return [st]

            if isinstance(st, PyIRFunctionDef):
                st.decorators = [rw_expr(d) for d in st.decorators]  # type: ignore[list-item]
                st.body = rw_stmt_list(st.body)
                return [st]

            if isinstance(st, PyIRFunctionExpr):
                st.body = rw_stmt_list(st.body)
                return [st]

            if isinstance(st, PyIRClassDef):
                st.bases = [rw_expr(b) for b in st.bases]
                st.decorators = [rw_expr(d) for d in st.decorators]  # type: ignore[list-item]
                st.body = rw_stmt_list(st.body)
                return [st]

            if isinstance(
                st,
                (
                    PyIRImport,
                    PyIRBreak,
                    PyIRContinue,
                    PyIRPass,
                    PyIRComment,
                    PyIRVarCreate,
                    PyIRConstant,
                    PyIRVarUse,
                    PyIRSymLink,
                ),
            ):
                return [st]

            return [rw_expr(st)]

        ir.body = rw_stmt_list(ir.body)
        return ir

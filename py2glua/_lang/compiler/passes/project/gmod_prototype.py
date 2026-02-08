from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    FileRealm,
    PyIRAnnotatedAssign,
    PyIRAssign,
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRVarCreate,
    PyIRVarUse,
)
from ..analysis.symlinks import PyIRSymLink, SymLinkContext
from ..common import (
    canon_path,
    collect_core_compiler_directive_local_symbol_ids,
    decorator_call_parts,
    is_expr_on_symbol_ids,
)


@dataclass(slots=True)
class _ProtoClass:
    class_name: str
    fold: str
    decl_path: Path | None
    decl_node: PyIRNode

    fields_by_realm: dict[str, dict[str, PyIRNode]]
    methods_by_realm: dict[str, dict[str, PyIRFunctionDef]]


@dataclass(slots=True)
class _ProtoInjectedMethod:
    kind: str
    realm: str
    method_name: str
    fn: PyIRFunctionDef
    at_path: Path | None
    at_node: PyIRNode


@dataclass(slots=True)
class _ProtoInstance:
    var_name: str
    uid: str
    fold: str
    global_name: str
    decl_path: Path | None
    decl_node: PyIRNode

    proto: _ProtoClass

    fields_by_realm: dict[str, dict[str, PyIRNode]]
    injected_methods: list[_ProtoInjectedMethod]


class BuildGmodPrototypesProjectPass:
    """
    Project-pass: реализует CompilerDirective.gmod_prototype(fold).

    Поля:
      - Эмитим только если у поля есть value (ann-only игнорим)
      - Учитываем realm: if SERVER/CLIENT/MENU/SHARED или if Realm.SERVER и т.п.
      - uid никогда не эмитим

    Методы:
      - Эмитим если тело не "pass"
      - @gmod_api не эмитим (страховка)
      - DSL методы override/add_method не эмитим как обычные методы класса

    Инстансы:
      - x = ENT() / x = ENT.__init__()
      - x.uid = "..." меняет uid папки
      - x.<field> = <expr> добавляет поле в realm по месту
      - @x.override(realm, ENT.Method) заменяет метод в realm
      - @x.add_method(realm) добавляет метод в realm (по имени функции), без замены, с проверкой коллизий

    Генерим:
      lua/<fold>/<uid>/shared.py
      lua/<fold>/<uid>/init.py
      lua/<fold>/<uid>/cl_init.py

    И режем DSL из исходников:
      - prototype classdef
      - инстанс-DSL / override / add_method / присваивания полей инстанса
    """

    _DEC_GMOD_PROTO: Final[str] = "gmod_prototype"
    _DEC_OVERRIDE: Final[str] = "override"
    _DEC_ADD_METHOD: Final[str] = "add_method"
    _DEC_GMOD_API: Final[str] = "gmod_api"

    _REALMS: Final[set[str]] = {"SERVER", "CLIENT", "SHARED", "MENU"}

    @classmethod
    def run(cls, files: list[PyIRFile], ctx: SymLinkContext) -> list[PyIRFile]:
        ctx.ensure_module_index()

        core_cd_symids_by_file: dict[Path, set[int]] = {}
        for ir in files:
            if ir.path is None:
                continue
            current_module = ctx.module_name_by_path.get(ir.path)
            if not current_module:
                continue
            core_cd_symids_by_file[ir.path] = (
                collect_core_compiler_directive_local_symbol_ids(
                    ir,
                    ctx=ctx,
                    current_module=current_module,
                )
            )

        proto_by_decl: dict[tuple[str, str], _ProtoClass] = {}
        for ir in files:
            if ir.path is None:
                continue
            core_cd_symids = core_cd_symids_by_file.get(ir.path, set())

            for st in ir.body:
                if not isinstance(st, PyIRClassDef):
                    continue

                fold = cls._extract_gmod_prototype_fold(
                    n=st,
                    ir=ir,
                    core_cd_symids=core_cd_symids,
                    ctx=ctx,
                )
                if fold is None:
                    continue

                fields_by_realm, methods_by_realm = cls._collect_class_members(
                    st,
                    core_cd_symids=core_cd_symids,
                    ctx=ctx,
                )

                canon = canon_path(ir.path)
                if canon is None:
                    continue

                key = (canon.as_posix(), st.name)
                proto_by_decl[key] = _ProtoClass(
                    class_name=st.name,
                    fold=fold,
                    decl_path=ir.path,
                    decl_node=st,
                    fields_by_realm=fields_by_realm,
                    methods_by_realm=methods_by_realm,
                )

        if not proto_by_decl:
            return files

        instances: list[_ProtoInstance] = []
        stripped_files: list[PyIRFile] = []

        for ir in files:
            if ir.path is None:
                stripped_files.append(ir)
                continue

            inst_by_var: dict[str, _ProtoInstance] = {}
            removed_func_names: set[str] = set()

            ir.body = cls._strip_dsl_in_block(
                body=list(ir.body),
                ir=ir,
                proto_by_decl=proto_by_decl,
                core_cd_symids=core_cd_symids_by_file.get(ir.path, set()),
                ctx=ctx,
                inst_by_var=inst_by_var,
                instances_out=instances,
                realm_ctx="SHARED",
                removed_func_names=removed_func_names,
            )
            stripped_files.append(ir)

        seen: dict[tuple[str, str], _ProtoInstance] = {}
        for inst in instances:
            key = (inst.fold, inst.uid)
            if key in seen:
                other = seen[key]
                CompilerExit.user_error_node(
                    f"gmod_prototype: uid коллизия: fold={inst.fold!r} uid={inst.uid!r}",
                    path=inst.decl_path,
                    node=inst.decl_node,
                )
                CompilerExit.user_error_node(
                    "gmod_prototype: первое объявление этого uid было тут",
                    path=other.decl_path,
                    node=other.decl_node,
                )
            seen[key] = inst

        generated: list[PyIRFile] = []
        for inst in instances:
            generated.extend(cls._emit_instance(inst))

        return [*generated, *stripped_files]

    @classmethod
    def _strip_dsl_in_block(
        cls,
        *,
        body: list[PyIRNode],
        ir: PyIRFile,
        proto_by_decl: dict[tuple[str, str], _ProtoClass],
        core_cd_symids: set[int],
        ctx: SymLinkContext,
        inst_by_var: dict[str, _ProtoInstance],
        instances_out: list[_ProtoInstance],
        realm_ctx: str,
        removed_func_names: set[str],
    ) -> list[PyIRNode]:
        out: list[PyIRNode] = []

        for st in body:
            if isinstance(st, PyIRIf):
                r = cls._realm_from_test(st.test)
                if r is not None and r in cls._REALMS:
                    st.body = cls._strip_dsl_in_block(
                        body=list(st.body or []),
                        ir=ir,
                        proto_by_decl=proto_by_decl,
                        core_cd_symids=core_cd_symids,
                        ctx=ctx,
                        inst_by_var=inst_by_var,
                        instances_out=instances_out,
                        realm_ctx=r,
                        removed_func_names=removed_func_names,
                    )
                    st.orelse = cls._strip_dsl_in_block(
                        body=list(st.orelse or []),
                        ir=ir,
                        proto_by_decl=proto_by_decl,
                        core_cd_symids=core_cd_symids,
                        ctx=ctx,
                        inst_by_var=inst_by_var,
                        instances_out=instances_out,
                        realm_ctx=realm_ctx,
                        removed_func_names=removed_func_names,
                    )
                else:
                    st.body = cls._strip_dsl_in_block(
                        body=list(st.body or []),
                        ir=ir,
                        proto_by_decl=proto_by_decl,
                        core_cd_symids=core_cd_symids,
                        ctx=ctx,
                        inst_by_var=inst_by_var,
                        instances_out=instances_out,
                        realm_ctx=realm_ctx,
                        removed_func_names=removed_func_names,
                    )
                    st.orelse = cls._strip_dsl_in_block(
                        body=list(st.orelse or []),
                        ir=ir,
                        proto_by_decl=proto_by_decl,
                        core_cd_symids=core_cd_symids,
                        ctx=ctx,
                        inst_by_var=inst_by_var,
                        instances_out=instances_out,
                        realm_ctx=realm_ctx,
                        removed_func_names=removed_func_names,
                    )

                if not (st.body or []) and not (st.orelse or []):
                    continue

                out.append(st)
                continue

            if isinstance(st, PyIRClassDef) and cls._is_gmod_prototype_classdef(
                st,
                ir=ir,
                core_cd_symids=core_cd_symids,
                ctx=ctx,
            ):
                continue

            if isinstance(st, PyIRVarCreate) and st.name in removed_func_names:
                continue

            hit = cls._match_instance_create(st=st, ir=ir, proto_by_decl=proto_by_decl)
            if hit is not None:
                inst_by_var[hit.var_name] = hit
                instances_out.append(hit)
                continue

            if isinstance(st, PyIRAssign) and len(st.targets) == 1:
                t = st.targets[0]
                if isinstance(t, PyIRAttribute):
                    base_name = cls._extract_name(t.value)
                    if base_name and base_name in inst_by_var:
                        inst = inst_by_var[base_name]
                        field = t.attr

                        if field == "uid":
                            uid = cls._require_string_literal(
                                st.value, path=ir.path, node=st
                            )
                            if uid.strip():
                                inst.uid = uid.strip()
                            continue

                        inst.fields_by_realm.setdefault(realm_ctx, {})[field] = st.value
                        continue

            if isinstance(st, PyIRFunctionDef):
                inj = cls._extract_injected_method(st, ir=ir, inst_by_var=inst_by_var)
                if inj is not None:
                    inst, injected = inj
                    inst.injected_methods.append(injected)
                    removed_func_names.add(st.name)
                    continue

            out.append(st)

        return out

    @classmethod
    def _extract_gmod_prototype_fold(
        cls,
        *,
        n: PyIRClassDef,
        ir: PyIRFile,
        core_cd_symids: set[int],
        ctx: SymLinkContext,
    ) -> str | None:
        for d in n.decorators or []:
            func, args_p, args_kw = decorator_call_parts(d)

            if not isinstance(func, PyIRAttribute):
                continue
            if func.attr != cls._DEC_GMOD_PROTO:
                continue
            if not is_expr_on_symbol_ids(
                func.value,
                symbol_ids=core_cd_symids,
                ctx=ctx,
            ):
                continue

            fold_node: PyIRNode | None = None

            if args_kw:
                fold_node = args_kw.get("fold")
                if fold_node is None:
                    CompilerExit.user_error_node(
                        "CompilerDirective.gmod_prototype: ожидается аргумент fold",
                        path=ir.path,
                        node=d,
                    )
            elif args_p:
                if len(args_p) != 1:
                    CompilerExit.user_error_node(
                        "CompilerDirective.gmod_prototype: ожидается ровно 1 позиционный аргумент (fold)",
                        path=ir.path,
                        node=d,
                    )
                fold_node = args_p[0]
            else:
                CompilerExit.user_error_node(
                    "CompilerDirective.gmod_prototype: ожидается аргумент fold",
                    path=ir.path,
                    node=d,
                )

            if not isinstance(fold_node, PyIRConstant) or not isinstance(
                fold_node.value, str
            ):
                CompilerExit.user_error_node(
                    "CompilerDirective.gmod_prototype: fold должен быть строковым литералом",
                    path=ir.path,
                    node=d,
                )

            fold = fold_node.value.strip().replace("\\", "/").strip("/")
            if not fold:
                CompilerExit.user_error_node(
                    "CompilerDirective.gmod_prototype: fold не может быть пустым",
                    path=ir.path,
                    node=d,
                )

            return fold

        return None

    @classmethod
    def _is_gmod_prototype_classdef(
        cls,
        n: PyIRClassDef,
        *,
        ir: PyIRFile,
        core_cd_symids: set[int],
        ctx: SymLinkContext,
    ) -> bool:
        return (
            cls._extract_gmod_prototype_fold(
                n=n,
                ir=ir,
                core_cd_symids=core_cd_symids,
                ctx=ctx,
            )
            is not None
        )

    @classmethod
    def _collect_class_members(
        cls,
        n: PyIRClassDef,
        *,
        core_cd_symids: set[int],
        ctx: SymLinkContext,
    ) -> tuple[dict[str, dict[str, PyIRNode]], dict[str, dict[str, PyIRFunctionDef]]]:
        fields_by_realm: dict[str, dict[str, PyIRNode]] = {}
        methods_by_realm: dict[str, dict[str, PyIRFunctionDef]] = {}

        def add_field(realm: str, name: str, value: PyIRNode) -> None:
            if not name or name == "uid":
                return
            fields_by_realm.setdefault(realm, {})[name] = value

        def add_method(realm: str, fn: PyIRFunctionDef) -> None:
            if not fn.name:
                return

            if fn.name in (cls._DEC_OVERRIDE, cls._DEC_ADD_METHOD):
                return
            if cls._is_gmod_api_method(fn, core_cd_symids=core_cd_symids, ctx=ctx):
                return
            if cls._is_effectively_pass(fn):
                return
            methods_by_realm.setdefault(realm, {})[fn.name] = fn

        def try_extract_field_from_stmt(st: PyIRNode, realm_ctx: str) -> bool:

            if isinstance(st, PyIRAssign):
                for t in st.targets:
                    name = cls._extract_name(t)
                    if name:
                        add_field(realm_ctx, name, st.value)
                return True

            if isinstance(st, PyIRAnnotatedAssign):
                val = st.value
                if isinstance(val, PyIRNode):
                    add_field(realm_ctx, st.name, val)
                return True

            if isinstance(st, PyIRVarCreate):
                return True

            return False

        def walk(body: list[PyIRNode], realm_ctx: str) -> None:
            for st in body:
                if isinstance(st, PyIRIf):
                    r = cls._realm_from_test(st.test)
                    if r is not None and r in cls._REALMS:
                        walk(list(st.body or []), r)
                        walk(list(st.orelse or []), realm_ctx)
                    else:
                        walk(list(st.body or []), realm_ctx)
                        walk(list(st.orelse or []), realm_ctx)
                    continue

                if try_extract_field_from_stmt(st, realm_ctx):
                    continue

                if isinstance(st, PyIRFunctionDef):
                    add_method(realm_ctx, st)
                    continue

        walk(list(n.body or []), "SHARED")
        return fields_by_realm, methods_by_realm

    @classmethod
    def _realm_from_test(cls, test: PyIRNode) -> str | None:
        name: str | None = None

        if isinstance(test, PyIRAttribute):
            name = test.attr
        elif isinstance(test, PyIRSymLink):
            name = test.name
        elif isinstance(test, PyIRVarUse):
            name = test.name
        elif isinstance(test, PyIREmitExpr) and test.kind == PyIREmitKind.GLOBAL:
            name = test.name
        elif isinstance(test, PyIRConstant) and isinstance(test.value, str):
            name = test.value

        if not isinstance(name, str) or not name:
            return None

        cand = name.strip().upper()
        if cand in cls._REALMS:
            return cand
        return None

    @classmethod
    def _is_gmod_api_method(
        cls,
        fn: PyIRFunctionDef,
        *,
        core_cd_symids: set[int],
        ctx: SymLinkContext,
    ) -> bool:
        for d in fn.decorators or []:
            func, _args_p, _args_kw = decorator_call_parts(d)
            if not isinstance(func, PyIRAttribute):
                continue
            if func.attr != cls._DEC_GMOD_API:
                continue
            if is_expr_on_symbol_ids(
                func.value,
                symbol_ids=core_cd_symids,
                ctx=ctx,
            ):
                return True
        return False

    @staticmethod
    def _is_effectively_pass(fn: PyIRFunctionDef) -> bool:
        body = list(fn.body or [])
        if not body:
            return True

        for st in body:
            tname = type(st).__name__
            if tname == "PyIRPass":
                continue
            if isinstance(st, PyIREmitExpr) and st.kind == PyIREmitKind.RAW:
                s = (st.name or "").strip().lower()
                if s in ("", "pass", "-- pass"):
                    continue
            if isinstance(st, PyIRConstant) and (st.value is Ellipsis):
                continue
            return False

        return True

    @classmethod
    def _match_instance_create(
        cls,
        *,
        st: PyIRNode,
        ir: PyIRFile,
        proto_by_decl: dict[tuple[str, str], _ProtoClass],
    ) -> _ProtoInstance | None:
        if not isinstance(st, PyIRAssign):
            return None
        if len(st.targets) != 1:
            return None

        var_name = cls._extract_name(st.targets[0])
        if not var_name:
            return None

        proto = cls._extract_called_proto(st.value, proto_by_decl)
        if proto is None:
            return None

        uid = var_name

        return _ProtoInstance(
            var_name=var_name,
            uid=uid,
            fold=proto.fold,
            global_name=proto.class_name,
            decl_path=ir.path,
            decl_node=st,
            proto=proto,
            fields_by_realm={},
            injected_methods=[],
        )

    @classmethod
    def _extract_called_proto(
        cls,
        expr: PyIRNode,
        proto_by_decl: dict[tuple[str, str], _ProtoClass],
    ) -> _ProtoClass | None:
        if not isinstance(expr, PyIRCall):
            return None

        p = cls._proto_from_callable(expr.func, proto_by_decl)
        if p is not None:
            return p

        if isinstance(expr.func, PyIRAttribute) and expr.func.attr == "__init__":
            return cls._proto_from_callable(expr.func.value, proto_by_decl)

        return None

    @classmethod
    def _proto_from_callable(
        cls,
        callee: PyIRNode,
        proto_by_decl: dict[tuple[str, str], _ProtoClass],
    ) -> _ProtoClass | None:
        if isinstance(callee, PyIRSymLink):
            if callee.decl_file is None:
                return None
            decl_path = canon_path(callee.decl_file)
            if decl_path is None:
                return None
            key = (decl_path.as_posix(), callee.name)
            return proto_by_decl.get(key)

        if isinstance(callee, PyIRVarUse):
            name = callee.name
            if name:
                for (_p, n), pc in proto_by_decl.items():
                    if n == name:
                        return pc

        return None

    @classmethod
    def _extract_injected_method(
        cls,
        fn: PyIRFunctionDef,
        *,
        ir: PyIRFile,
        inst_by_var: dict[str, _ProtoInstance],
    ) -> tuple[_ProtoInstance, _ProtoInjectedMethod] | None:
        for d in fn.decorators or []:
            func, args_p, args_kw = decorator_call_parts(d)

            if args_kw:
                CompilerExit.user_error_node(
                    "override/add_method: keyword-аргументы не поддерживаются, используй позиционные",
                    path=ir.path,
                    node=d,
                )

            if not isinstance(func, PyIRAttribute):
                continue

            base_name = cls._extract_name(func.value)
            if not base_name:
                continue
            inst = inst_by_var.get(base_name)
            if inst is None:
                continue

            if func.attr == cls._DEC_OVERRIDE:
                if len(args_p) != 2:
                    CompilerExit.user_error_node(
                        "override: ожидается 2 аргумента (realm, target)",
                        path=ir.path,
                        node=d,
                    )
                realm = cls._extract_realm(args_p[0], ir.path, d)
                method_name = cls._extract_target_method(args_p[1], ir.path, d)
                return (
                    inst,
                    _ProtoInjectedMethod(
                        kind="override",
                        realm=realm,
                        method_name=method_name,
                        fn=fn,
                        at_path=ir.path,
                        at_node=d,
                    ),
                )

            if func.attr == cls._DEC_ADD_METHOD:
                if len(args_p) != 1:
                    CompilerExit.user_error_node(
                        "add_method: ожидается 1 аргумент (realm)",
                        path=ir.path,
                        node=d,
                    )
                realm = cls._extract_realm(args_p[0], ir.path, d)
                if not fn.name:
                    CompilerExit.user_error_node(
                        "add_method: у функции должно быть имя",
                        path=ir.path,
                        node=fn,
                    )
                return (
                    inst,
                    _ProtoInjectedMethod(
                        kind="add",
                        realm=realm,
                        method_name=fn.name,
                        fn=fn,
                        at_path=ir.path,
                        at_node=d,
                    ),
                )

        return None

    @classmethod
    def _extract_target_method(
        cls, expr: PyIRNode, path: Path | None, node: PyIRNode
    ) -> str:
        if isinstance(expr, PyIRAttribute):
            if not expr.attr:
                CompilerExit.user_error_node(
                    "override: target должен быть вида Class.Method",
                    path=path,
                    node=node,
                )
            return expr.attr

        if isinstance(expr, PyIRSymLink) and expr.name:
            return expr.name

        CompilerExit.user_error_node(
            "override: target должен быть вида Class.Method",
            path=path,
            node=node,
        )
        assert False, "unreachable"

    @classmethod
    def _extract_realm(cls, expr: PyIRNode, path: Path | None, node: PyIRNode) -> str:
        r = cls._realm_from_test(expr)
        if r is not None and r in cls._REALMS:
            return r

        CompilerExit.user_error_node(
            "realm должен быть одним из SERVER|CLIENT|SHARED|MENU",
            path=path,
            node=node,
        )
        assert False, "unreachable"

    @classmethod
    def _emit_instance(cls, inst: _ProtoInstance) -> list[PyIRFile]:
        uid = inst.uid.strip()
        if not uid:
            CompilerExit.user_error_node(
                "gmod_prototype: uid пустой",
                path=inst.decl_path,
                node=inst.decl_node,
            )

        fold_path = Path(*inst.fold.split("/"))
        base_dir = Path("lua") / fold_path / uid

        shared_body = cls._build_realm_body(inst, out_realm="SHARED")
        init_body = cls._build_realm_body(inst, out_realm="SERVER")
        cl_body = cls._build_realm_body(inst, out_realm="CLIENT")

        shared = PyIRFile(
            line=None,
            offset=None,
            path=base_dir / "shared.py",
            body=shared_body if shared_body else [cls._raw("")],
            imports=[],
            realm=FileRealm.SHARED,
        )

        initf = PyIRFile(
            line=None,
            offset=None,
            path=base_dir / "init.py",
            body=(
                [
                    cls._raw('AddCSLuaFile("shared.lua")'),
                    cls._raw('AddCSLuaFile("cl_init.lua")'),
                    cls._raw('include("shared.lua")'),
                ]
                + (init_body if init_body else [cls._raw("")])
            ),  # type: ignore
            imports=[],
            realm=FileRealm.SERVER,
        )

        clf = PyIRFile(
            line=None,
            offset=None,
            path=base_dir / "cl_init.py",
            body=(
                [cls._raw('include("shared.lua")')]
                + (cl_body if cl_body else [cls._raw("")])
            ),  # type: ignore
            imports=[],
            realm=FileRealm.CLIENT,
        )

        return [shared, initf, clf]

    @classmethod
    def _build_realm_body(
        cls, inst: _ProtoInstance, *, out_realm: str
    ) -> list[PyIRNode]:
        g = inst.global_name
        out: list[PyIRNode] = []

        in_realms: tuple[str, ...]
        if out_realm == "CLIENT":
            in_realms = ("CLIENT", "MENU")
        else:
            in_realms = (out_realm,)

        base_fields: dict[str, PyIRNode] = {}
        base_methods: dict[str, PyIRFunctionDef] = {}

        for r in in_realms:
            base_fields.update(inst.proto.fields_by_realm.get(r, {}))
            base_methods.update(inst.proto.methods_by_realm.get(r, {}))

        inst_fields: dict[str, PyIRNode] = {}
        for r in in_realms:
            inst_fields.update(inst.fields_by_realm.get(r, {}))

        repl: dict[str, PyIRFunctionDef] = {}
        adds: dict[str, PyIRFunctionDef] = {}

        def belongs(inj_realm: str) -> bool:
            return inj_realm in in_realms

        for inj in inst.injected_methods:
            if not belongs(inj.realm):
                continue

            if inj.kind == "override":
                if inj.method_name in repl:
                    CompilerExit.user_error_node(
                        f"override: повторная перезапись метода {inj.method_name!r} в realm={inj.realm}",
                        path=inj.at_path,
                        node=inj.at_node,
                    )

                repl[inj.method_name] = inj.fn
                continue

            if inj.kind == "add":
                if inj.method_name in base_methods:
                    CompilerExit.user_error_node(
                        f"add_method: коллизия с базовым методом {inj.method_name!r} в realm={inj.realm}",
                        path=inj.at_path,
                        node=inj.at_node,
                    )
                if inj.method_name in repl:
                    CompilerExit.user_error_node(
                        f"add_method: коллизия с override-методом {inj.method_name!r} в realm={inj.realm}",
                        path=inj.at_path,
                        node=inj.at_node,
                    )
                if inj.method_name in adds:
                    CompilerExit.user_error_node(
                        f"add_method: повторное добавление метода {inj.method_name!r} в realm={inj.realm}",
                        path=inj.at_path,
                        node=inj.at_node,
                    )
                adds[inj.method_name] = inj.fn
                continue

            CompilerExit.user_error_node(
                f"internal: неизвестный тип injected-метода {inj.kind!r}",
                path=inj.at_path,
                node=inj.at_node,
            )
            assert False, "unreachable"

        merged_fields = dict(base_fields)
        merged_fields.update(inst_fields)

        for k in sorted(merged_fields.keys()):
            if k == "uid":
                continue

            out.append(
                PyIRAssign(
                    line=None,
                    offset=None,
                    targets=[
                        PyIRAttribute(
                            line=None, offset=None, value=cls._glob(g), attr=k
                        )
                    ],
                    value=merged_fields[k],
                )
            )

        for name in sorted(base_methods.keys()):
            if name in repl:
                continue

            out.append(cls._emit_method(g, name, base_methods[name]))

        for name in sorted(repl.keys()):
            out.append(cls._emit_method(g, name, repl[name]))

        for name in sorted(adds.keys()):
            out.append(cls._emit_method(g, name, adds[name]))

        return out

    @classmethod
    def _emit_method(
        cls, g: str, method_name: str, src: PyIRFunctionDef
    ) -> PyIRFunctionDef:
        return PyIRFunctionDef(
            line=src.line,
            offset=src.offset,
            name=f"{g}.{method_name}",
            signature=dict(src.signature),
            returns=src.returns,
            vararg=src.vararg,
            kwarg=src.kwarg,
            decorators=[],
            body=list(src.body),
        )

    @staticmethod
    def _extract_name(n: PyIRNode) -> str | None:
        if isinstance(n, PyIRSymLink):
            return n.name

        if isinstance(n, PyIRVarUse):
            return n.name

        if isinstance(n, PyIREmitExpr) and n.kind == PyIREmitKind.GLOBAL:
            return n.name

        return None

    @staticmethod
    def _require_string_literal(
        expr: PyIRNode, *, path: Path | None, node: PyIRNode
    ) -> str:
        if isinstance(expr, PyIRConstant) and isinstance(expr.value, str):
            return expr.value

        CompilerExit.user_error_node(
            "Ожидается строковый литерал", path=path, node=node
        )

        assert False, "unreachable"

    @staticmethod
    def _raw(s: str) -> PyIREmitExpr:
        return PyIREmitExpr(line=None, offset=None, kind=PyIREmitKind.RAW, name=s)

    @staticmethod
    def _glob(name: str) -> PyIREmitExpr:
        return PyIREmitExpr(line=None, offset=None, kind=PyIREmitKind.GLOBAL, name=name)

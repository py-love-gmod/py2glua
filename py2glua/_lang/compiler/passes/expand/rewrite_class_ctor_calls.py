from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Literal

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRAnnotatedAssign,
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRDecorator,
    PyIRDict,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRList,
    PyIRNode,
    PyIRReturn,
    PyIRTuple,
    PyIRVarCreate,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
)
from ..rewrite_utils import rewrite_expr_with_store_default, rewrite_stmt_block
from .expand_context import ClassTemplate, ExpandContext, FnSig

MethodKind = Literal["instance", "class", "static"]
ReceiverEnv = dict[str, tuple[str, bool]]


def _file_key(path: Path | None) -> str:
    if path is None:
        return "<none>"
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _decorator_name(dec: PyIRDecorator) -> str | None:
    expr: PyIRNode = dec.exper
    if isinstance(expr, PyIRCall):
        expr = expr.func

    if isinstance(expr, PyIRVarUse):
        return expr.name

    if isinstance(expr, PyIRAttribute):
        return expr.attr

    return None


def _classify_method_kind(
    ir: PyIRFile,
    fn: PyIRFunctionDef,
) -> MethodKind:
    has_classmethod = False
    has_staticmethod = False

    for d in fn.decorators:
        name = _decorator_name(d)
        if name == "classmethod":
            has_classmethod = True
        elif name == "staticmethod":
            has_staticmethod = True

    if has_classmethod and has_staticmethod:
        CompilerExit.user_error_node(
            "Метод не может быть одновременно @classmethod и @staticmethod",
            ir.path,
            fn,
        )

    if has_classmethod:
        return "class"
    if has_staticmethod:
        return "static"
    return "instance"


def _strip_method_kind_decorators(
    decorators: list[PyIRDecorator],
) -> list[PyIRDecorator]:
    out: list[PyIRDecorator] = []
    for d in decorators:
        name = _decorator_name(d)
        if name in {"classmethod", "staticmethod"}:
            continue
        out.append(d)
    return out


def _collect_binding_names(target: PyIRNode, out: set[str]) -> None:
    if isinstance(target, PyIRVarUse):
        out.add(target.name)
        return

    if isinstance(target, (PyIRTuple, PyIRList)):
        for el in target.elements:
            _collect_binding_names(el, out)


class CollectClassRuntimePass:
    """
    Собирает пофайловые шаблоны классов для последующих rewrite-пассов:
      - список instance-методов (с их сигнатурами),
      - наличие user-__init__.
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> None:
        file_key = _file_key(ir.path)
        templates: Dict[str, ClassTemplate] = {}

        for st in ir.body:
            if not isinstance(st, PyIRClassDef):
                continue

            instance_methods: Dict[str, FnSig] = {}
            class_methods: list[str] = []

            for item in st.body:
                if not isinstance(item, PyIRFunctionDef):
                    continue

                kind = _classify_method_kind(ir, item)
                if kind == "class":
                    class_methods.append(item.name)
                    continue

                if kind != "instance":
                    continue

                defaults: Dict[str, PyIRNode] = {}
                for p_name, (_ann, default) in item.signature.items():
                    if default is not None:
                        defaults[p_name] = default

                instance_methods[item.name] = FnSig(
                    params=tuple(item.signature.keys()),
                    defaults=defaults,
                    vararg_name=item.vararg,
                    kwarg_name=item.kwarg,
                )

            templates[st.name] = ClassTemplate(
                instance_methods=instance_methods,
                class_methods=tuple(class_methods),
                has_init=("__init__" in instance_methods),
            )

        ctx.class_templates[file_key] = templates


class RewriteClassMethodDecoratorsPass:
    """
    Валидирует конфликт @classmethod/@staticmethod и удаляет эти декораторы.
    Семантика передачи первого аргумента обрабатывается отдельным rewrite-pass.
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        _ = ctx

        for st in ir.body:
            if not isinstance(st, PyIRClassDef):
                continue

            RewriteClassMethodDecoratorsPass._rewrite_class_body(ir, st)

        return ir

    @staticmethod
    def _rewrite_class_body(ir: PyIRFile, cls: PyIRClassDef) -> None:
        for item in cls.body:
            if not isinstance(item, PyIRFunctionDef):
                continue

            _classify_method_kind(ir, item)
            item.decorators = _strip_method_kind_decorators(item.decorators)


class RewriteClassCtorCallsPass:
    """
    Нормализует конструирование классов в 2 шага:
      1) Внутри каждого класса строит runtime-конструктор `__init__`, который:
         - создаёт объект через setmetatable({}, Class),

         - вызывает user-init (если он был),
         - возвращает объект.
      2) Переписывает вызовы `Class(...)` в `Class.__init__(...)`.
    """

    _CTOR_NAME = "__init__"

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        templates = ctx.class_templates.get(_file_key(ir.path), {})
        if not templates:
            return ir

        for st in ir.body:
            if not isinstance(st, PyIRClassDef):
                continue
            template = templates.get(st.name)
            if template is None:
                continue
            RewriteClassCtorCallsPass._ensure_runtime_ctor(
                ir=ir,
                cls=st,
                template=template,
            )

        local_classes = set(templates.keys())

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(expr: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(expr, PyIRCall):
                expr.func = rw_expr(expr.func, False)
                expr.args_p = [rw_expr(a, False) for a in expr.args_p]
                expr.args_kw = {k: rw_expr(v, False) for k, v in expr.args_kw.items()}

                if not store and isinstance(expr.func, PyIRVarUse):
                    if expr.func.name in local_classes:
                        if expr.args_kw:
                            CompilerExit.user_error_node(
                                f"Вызов конструктора класса '{expr.func.name}' с keyword-аргументами пока не поддерживается",
                                ir.path,
                                expr,
                            )
                        base = expr.func
                        expr.func = PyIRAttribute(
                            line=base.line,
                            offset=base.offset,
                            value=base,
                            attr=RewriteClassCtorCallsPass._CTOR_NAME,
                        )

                return expr

            return rewrite_expr_with_store_default(
                expr,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

    @staticmethod
    def _ensure_runtime_ctor(
        *,
        ir: PyIRFile,
        cls: PyIRClassDef,
        template: ClassTemplate,
    ) -> None:
        user_init: PyIRFunctionDef | None = None
        user_init_idx: int | None = None

        for i, item in enumerate(cls.body):
            if not isinstance(item, PyIRFunctionDef):
                continue
            if item.name != RewriteClassCtorCallsPass._CTOR_NAME:
                continue
            if _classify_method_kind(ir, item) != "instance":
                continue

            user_init = item
            user_init_idx = i
            break

        ctor = RewriteClassCtorCallsPass._build_runtime_ctor(
            class_name=cls.name,
            template=template,
            user_init=user_init,
        )

        if user_init_idx is not None:
            cls.body[user_init_idx] = ctor
        else:
            cls.body.append(ctor)

    @staticmethod
    def _build_runtime_ctor(
        *,
        class_name: str,
        template: ClassTemplate,
        user_init: PyIRFunctionDef | None,
    ) -> PyIRFunctionDef:
        _ = template
        obj_name = "__obj"

        wrapper_sig: dict[str, tuple[str | None, PyIRNode | None]] = {}
        wrapper_vararg: str | None = None
        wrapper_kwarg: str | None = None

        line = user_init.line if user_init is not None else None
        offset = user_init.offset if user_init is not None else None

        init_self_name: str | None = None
        if user_init is not None:
            params = list(user_init.signature.items())
            if params:
                init_self_name = params[0][0]

            tail_params = params[1:] if params else []
            for p_name, (ann, default) in tail_params:
                wrapper_sig[p_name] = (
                    ann,
                    deepcopy(default) if default is not None else None,
                )

            wrapper_vararg = user_init.vararg
            wrapper_kwarg = user_init.kwarg

        body: list[PyIRNode] = [
            PyIRAssign(
                line=line,
                offset=offset,
                targets=[
                    PyIRVarUse(
                        line=line,
                        offset=offset,
                        name=obj_name,
                    )
                ],
                value=PyIRCall(
                    line=line,
                    offset=offset,
                    func=PyIRVarUse(
                        line=line,
                        offset=offset,
                        name="setmetatable",
                    ),
                    args_p=[
                        PyIRDict(line=line, offset=offset, items=[]),
                        PyIRVarUse(
                            line=line,
                            offset=offset,
                            name=class_name,
                        ),
                    ],
                    args_kw={},
                ),
            )
        ]

        if user_init is not None:
            for st in deepcopy(user_init.body):
                if init_self_name is not None and init_self_name != obj_name:
                    RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                        st,
                        src=init_self_name,
                        dst=obj_name,
                    )
                RewriteClassCtorCallsPass._rewrite_init_returns_to_obj(
                    st, obj_name=obj_name
                )
                body.append(st)

        body.append(
            PyIRReturn(
                line=line,
                offset=offset,
                value=PyIRVarUse(line=line, offset=offset, name=obj_name),
            )
        )

        return PyIRFunctionDef(
            line=line,
            offset=offset,
            name=RewriteClassCtorCallsPass._CTOR_NAME,
            signature=wrapper_sig,
            returns=(user_init.returns if user_init is not None else None),
            vararg=wrapper_vararg,
            kwarg=wrapper_kwarg,
            decorators=(deepcopy(user_init.decorators) if user_init is not None else []),
            body=body,
        )

    @staticmethod
    def _replace_var_uses_in_stmt(
        st: PyIRNode,
        *,
        src: str,
        dst: str,
    ) -> None:
        if isinstance(st, PyIRVarUse):
            if st.name == src:
                st.name = dst
            return

        if isinstance(st, (PyIRFunctionDef, PyIRClassDef)):
            return

        if isinstance(st, PyIRIf):
            RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                st.test, src=src, dst=dst
            )
            for child in st.body:
                RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                    child, src=src, dst=dst
                )
            for child in st.orelse:
                RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                    child, src=src, dst=dst
                )
            return

        if isinstance(st, PyIRWhile):
            RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                st.test, src=src, dst=dst
            )
            for child in st.body:
                RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                    child, src=src, dst=dst
                )
            return

        if isinstance(st, PyIRFor):
            RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                st.target, src=src, dst=dst
            )
            RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                st.iter, src=src, dst=dst
            )
            for child in st.body:
                RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                    child, src=src, dst=dst
                )
            return

        if isinstance(st, PyIRWith):
            for item in st.items:
                RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                    item.context_expr,
                    src=src,
                    dst=dst,
                )
                if item.optional_vars is not None:
                    RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                        item.optional_vars,
                        src=src,
                        dst=dst,
                    )
            for child in st.body:
                RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                    child, src=src, dst=dst
                )
            return

        def rw_expr(node: PyIRNode, _store: bool) -> PyIRNode:
            if isinstance(node, PyIRVarUse) and node.name == src:
                node.name = dst
                return node

            if isinstance(node, (PyIRFunctionDef, PyIRClassDef)):
                return node

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            for child in body:
                RewriteClassCtorCallsPass._replace_var_uses_in_stmt(
                    child, src=src, dst=dst
                )
            return body

        rewrite_expr_with_store_default(st, rw_expr=rw_expr, rw_block=rw_block)

    @staticmethod
    def _rewrite_init_returns_to_obj(st: PyIRNode, *, obj_name: str) -> None:
        if isinstance(st, PyIRReturn):
            st.value = PyIRVarUse(line=st.line, offset=st.offset, name=obj_name)
            return

        if isinstance(st, PyIRIf):
            for child in st.body:
                RewriteClassCtorCallsPass._rewrite_init_returns_to_obj(
                    child, obj_name=obj_name
                )
            for child in st.orelse:
                RewriteClassCtorCallsPass._rewrite_init_returns_to_obj(
                    child, obj_name=obj_name
                )
            return

        if isinstance(st, (PyIRWhile, PyIRFor, PyIRWith)):
            for child in st.body:
                RewriteClassCtorCallsPass._rewrite_init_returns_to_obj(
                    child, obj_name=obj_name
                )
            return

        if isinstance(st, (PyIRFunctionDef, PyIRClassDef)):
            return


class RewriteInstanceMethodCallsPass:
    """
    Переписывает вызовы instance-методов без `:` в явный вызов с self первым аргументом.

    Пример:
      foo = Test(...)
      foo.boo(1)   -> foo.boo(foo, 1)
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        templates = ctx.class_templates.get(_file_key(ir.path), {})
        if not templates:
            return ir

        ir.body = RewriteInstanceMethodCallsPass._rewrite_stmt_list(
            ir=ir,
            stmts=ir.body,
            templates=templates,
            env={},
            current_class=None,
        )
        return ir

    @staticmethod
    def _rewrite_stmt_list(
        *,
        ir: PyIRFile,
        stmts: list[PyIRNode],
        templates: dict[str, ClassTemplate],
        env: ReceiverEnv,
        current_class: str | None,
    ) -> list[PyIRNode]:
        return rewrite_stmt_block(
            stmts,
            lambda st: RewriteInstanceMethodCallsPass._rewrite_stmt(
                ir=ir,
                st=st,
                templates=templates,
                env=env,
                current_class=current_class,
            ),
        )

    @staticmethod
    def _rewrite_stmt(
        *,
        ir: PyIRFile,
        st: PyIRNode,
        templates: dict[str, ClassTemplate],
        env: ReceiverEnv,
        current_class: str | None,
    ) -> list[PyIRNode]:
        if isinstance(st, PyIRClassDef):
            env.pop(st.name, None)
            st.bases = [
                RewriteInstanceMethodCallsPass._rewrite_expr(
                    ir=ir,
                    expr=b,
                    templates=templates,
                    env=env,
                    current_class=current_class,
                )
                for b in st.bases
            ]
            fn_decorators: list[PyIRDecorator] = []
            for d in st.decorators:
                d_new = RewriteInstanceMethodCallsPass._rewrite_expr(
                    ir=ir,
                    expr=d,
                    templates=templates,
                    env=env,
                    current_class=current_class,
                )
                if not isinstance(d_new, PyIRDecorator):
                    CompilerExit.internal_error(
                        f"Ожидался тип PyIRDecorator, получен {type(d_new).__name__}"
                    )
                fn_decorators.append(d_new)
            st.decorators = fn_decorators
            st.body = RewriteInstanceMethodCallsPass._rewrite_stmt_list(
                ir=ir,
                stmts=st.body,
                templates=templates,
                env=dict(env),
                current_class=st.name if st.name in templates else None,
            )
            return [st]

        if isinstance(st, PyIRFunctionDef):
            env.pop(st.name, None)

            fn_env = dict(env)
            method_kind: MethodKind | None = None
            if current_class is not None:
                method_kind = RewriteInstanceMethodCallsPass._method_kind(
                    fn=st,
                    current_class=current_class,
                    templates=templates,
                )
            if (
                current_class is not None
                and method_kind in {"instance", "class"}
                and st.name != RewriteClassCtorCallsPass._CTOR_NAME
            ):
                first_param = next(iter(st.signature.keys()), None)
                if first_param is not None:
                    fn_env[first_param] = (
                        current_class,
                        method_kind == "instance",
                    )

            new_decorators: list[PyIRDecorator] = []
            for d in st.decorators:
                d_new = RewriteInstanceMethodCallsPass._rewrite_expr(
                    ir=ir,
                    expr=d,
                    templates=templates,
                    env=fn_env,
                    current_class=current_class,
                )
                if not isinstance(d_new, PyIRDecorator):
                    CompilerExit.internal_error(
                        f"Ожидался тип PyIRDecorator, получен {type(d_new).__name__}"
                    )
                new_decorators.append(d_new)
            st.decorators = new_decorators

            new_sig: dict[str, tuple[str | None, PyIRNode | None]] = {}
            for name, (ann, default) in st.signature.items():
                if default is not None:
                    default = RewriteInstanceMethodCallsPass._rewrite_expr(
                        ir=ir,
                        expr=default,
                        templates=templates,
                        env=fn_env,
                        current_class=current_class,
                    )
                new_sig[name] = (ann, default)
            st.signature = new_sig

            st.body = RewriteInstanceMethodCallsPass._rewrite_stmt_list(
                ir=ir,
                stmts=st.body,
                templates=templates,
                env=fn_env,
                current_class=None,
            )
            return [st]

        if isinstance(st, PyIRVarCreate):
            env.pop(st.name, None)
            return [st]

        if isinstance(st, PyIRAssign):
            st.value = RewriteInstanceMethodCallsPass._rewrite_expr(
                ir=ir,
                expr=st.value,
                templates=templates,
                env=env,
                current_class=current_class,
            )
            bound: set[str] = set()
            for t in st.targets:
                _collect_binding_names(t, bound)
            for n in bound:
                env.pop(n, None)

            instance_class = RewriteInstanceMethodCallsPass._infer_instance_class(
                expr=st.value,
                templates=templates,
                env=env,
            )
            if instance_class is not None:
                for n in bound:
                    env[n] = (instance_class, True)

            return [st]

        if isinstance(st, PyIRAnnotatedAssign):
            st.value = RewriteInstanceMethodCallsPass._rewrite_expr(
                ir=ir,
                expr=st.value,
                templates=templates,
                env=env,
                current_class=current_class,
            )
            env.pop(st.name, None)

            instance_class = RewriteInstanceMethodCallsPass._infer_instance_class(
                expr=st.value,
                templates=templates,
                env=env,
            )
            if instance_class is not None:
                env[st.name] = (instance_class, True)

            return [st]

        if isinstance(st, PyIRAugAssign):
            st.target = RewriteInstanceMethodCallsPass._rewrite_expr(
                ir=ir,
                expr=st.target,
                templates=templates,
                env=env,
                current_class=current_class,
            )
            st.value = RewriteInstanceMethodCallsPass._rewrite_expr(
                ir=ir,
                expr=st.value,
                templates=templates,
                env=env,
                current_class=current_class,
            )
            bound_aug: set[str] = set()
            _collect_binding_names(st.target, bound_aug)
            for n in bound_aug:
                env.pop(n, None)
            return [st]

        if isinstance(st, PyIRCall):
            return [
                RewriteInstanceMethodCallsPass._rewrite_expr(
                    ir=ir,
                    expr=st,
                    templates=templates,
                    env=env,
                    current_class=current_class,
                )
            ]

        return [
            RewriteInstanceMethodCallsPass._rewrite_expr(
                ir=ir,
                expr=st,
                templates=templates,
                env=env,
                current_class=current_class,
            )
        ]

    @staticmethod
    def _rewrite_expr(
        *,
        ir: PyIRFile,
        expr: PyIRNode,
        templates: dict[str, ClassTemplate],
        env: ReceiverEnv,
        current_class: str | None,
    ) -> PyIRNode:
        if isinstance(expr, PyIRCall):
            expr.func = RewriteInstanceMethodCallsPass._rewrite_expr(
                ir=ir,
                expr=expr.func,
                templates=templates,
                env=env,
                current_class=current_class,
            )
            expr.args_p = [
                RewriteInstanceMethodCallsPass._rewrite_expr(
                    ir=ir,
                    expr=a,
                    templates=templates,
                    env=env,
                    current_class=current_class,
                )
                for a in expr.args_p
            ]
            expr.args_kw = {
                k: RewriteInstanceMethodCallsPass._rewrite_expr(
                    ir=ir,
                    expr=v,
                    templates=templates,
                    env=env,
                    current_class=current_class,
                )
                for k, v in expr.args_kw.items()
            }

            RewriteInstanceMethodCallsPass._inject_method_receiver_arg_if_needed(
                expr=expr,
                templates=templates,
                env=env,
            )
            return expr

        def rw_expr(node: PyIRNode, _store: bool) -> PyIRNode:
            return RewriteInstanceMethodCallsPass._rewrite_expr(
                ir=ir,
                expr=node,
                templates=templates,
                env=env,
                current_class=current_class,
            )

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return RewriteInstanceMethodCallsPass._rewrite_stmt_list(
                ir=ir,
                stmts=body,
                templates=templates,
                env=dict(env),
                current_class=current_class,
            )

        return rewrite_expr_with_store_default(
            expr,
            rw_expr=rw_expr,
            rw_block=rw_block,
        )

    @staticmethod
    def _inject_method_receiver_arg_if_needed(
        *,
        expr: PyIRCall,
        templates: dict[str, ClassTemplate],
        env: ReceiverEnv,
    ) -> None:
        func = expr.func
        if not isinstance(func, PyIRAttribute):
            return

        receiver = func.value
        if not isinstance(receiver, PyIRVarUse):
            return

        receiver_class_name, receiver_is_instance = (
            RewriteInstanceMethodCallsPass._resolve_receiver_class(
                receiver=receiver,
                templates=templates,
                env=env,
            )
        )
        if receiver_class_name is None:
            return

        template = templates.get(receiver_class_name)
        if template is None:
            return

        if func.attr == RewriteClassCtorCallsPass._CTOR_NAME:
            return

        inject_name: str | None = None
        if receiver_is_instance and func.attr in template.instance_methods:
            inject_name = receiver.name

        if func.attr in template.class_methods:
            inject_name = receiver_class_name

        if inject_name is None:
            return

        if (
            expr.args_p
            and isinstance(expr.args_p[0], PyIRVarUse)
            and expr.args_p[0].name == inject_name
        ):
            return

        expr.args_p = [
            PyIRVarUse(
                line=receiver.line,
                offset=receiver.offset,
                name=inject_name,
            ),
            *expr.args_p,
        ]

    @staticmethod
    def _infer_instance_class(
        *,
        expr: PyIRNode,
        templates: dict[str, ClassTemplate],
        env: ReceiverEnv,
    ) -> str | None:
        if isinstance(expr, PyIRVarUse):
            mapped = env.get(expr.name)
            if mapped is None:
                return None
            class_name, is_instance = mapped
            if not is_instance:
                return None
            return class_name

        if not isinstance(expr, PyIRCall):
            return None

        if (
            isinstance(expr.func, PyIRVarUse)
            and expr.func.name == "setmetatable"
            and len(expr.args_p) >= 2
            and isinstance(expr.args_p[1], PyIRVarUse)
            and expr.args_p[1].name in templates
        ):
            return expr.args_p[1].name

        if not isinstance(expr.func, PyIRAttribute):
            return None

        if expr.func.attr != RewriteClassCtorCallsPass._CTOR_NAME:
            return None

        if not isinstance(expr.func.value, PyIRVarUse):
            return None

        class_name = expr.func.value.name
        if class_name not in templates:
            return None

        return class_name

    @staticmethod
    def _method_kind(
        *,
        fn: PyIRFunctionDef,
        current_class: str,
        templates: dict[str, ClassTemplate],
    ) -> MethodKind | None:
        template = templates.get(current_class)
        if template is None:
            return None

        if fn.name in template.class_methods:
            return "class"

        if fn.name in template.instance_methods:
            return "instance"

        return None

    @staticmethod
    def _resolve_receiver_class(
        *,
        receiver: PyIRVarUse,
        templates: dict[str, ClassTemplate],
        env: ReceiverEnv,
    ) -> tuple[str | None, bool]:
        mapped = env.get(receiver.name)
        if mapped is not None:
            return mapped

        if receiver.name in templates:
            return receiver.name, False

        return None, False

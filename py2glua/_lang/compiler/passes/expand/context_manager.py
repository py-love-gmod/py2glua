from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Dict, List, Tuple

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRDecorator,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
    PyIRVarUse,
    PyIRWith,
)
from .expand_context import ContextManagerTemplate, ExpandContext

_CD_PREFIXES = (
    (
        "py2glua",
        "glua",
        "core",
        "compiler_directive",
        "CompilerDirective",
    ),
    (
        "py2glua",
        "glua",
        "core",
        "CompilerDirective",
    ),
)
_CMDBODY_CHAINS = tuple(prefix + ("contextmanager_body",) for prefix in _CD_PREFIXES)


def _attr_chain_parts(node: PyIRNode) -> List[str] | None:
    names_rev: List[str] = []
    cur: PyIRNode = node

    while isinstance(cur, PyIRAttribute):
        names_rev.append(cur.attr)
        cur = cur.value

    if not isinstance(cur, PyIRVarUse):
        return None

    names_rev.append(cur.name)
    names_rev.reverse()
    return names_rev


def _is_compiler_directive_attr(attr: PyIRAttribute, *, method: str) -> bool:
    if attr.attr != method:
        return False

    parts = _attr_chain_parts(attr.value)
    if parts is None:
        return False

    return tuple(parts) in _CD_PREFIXES


def _compiler_directive_kind(dec: PyIRDecorator) -> str | None:
    expr = dec.exper

    if isinstance(expr, PyIRAttribute):
        if _is_compiler_directive_attr(expr, method="contextmanager"):
            return "contextmanager"
        return None

    if isinstance(expr, PyIRCall):
        f = expr.func
        if isinstance(f, PyIRAttribute):
            if _is_compiler_directive_attr(f, method="contextmanager"):
                return "contextmanager"
        return None

    return None


def _is_contextmanager_body_marker(node: PyIRNode) -> bool:
    if not isinstance(node, PyIRAttribute):
        return False

    parts = _attr_chain_parts(node)
    if parts is None:
        return False

    return tuple(parts) in _CMDBODY_CHAINS


def _get_param_names(fn: PyIRFunctionDef) -> Tuple[str, ...]:
    params = list(fn.signature.keys())
    if fn.vararg is not None:
        params.append(fn.vararg)
    if fn.kwarg is not None:
        params.append(fn.kwarg)
    return tuple(params)


def _split_contextmanager_fn(
    ir: PyIRFile, fn: PyIRFunctionDef
) -> ContextManagerTemplate:
    pre: List[PyIRNode] = []
    post: List[PyIRNode] = []

    seen = 0
    after = False

    for st in fn.body:
        if _is_contextmanager_body_marker(st):
            seen += 1
            after = True
            continue

        if not after:
            pre.append(st)
        else:
            post.append(st)

    if seen > 1:
        CompilerExit.user_error_node(
            "В contextmanager-функции marker 'CompilerDirective.contextmanager_body' встречается более одного раза",
            ir.path,
            fn,
        )

    insert_body = seen == 1

    return ContextManagerTemplate(
        pre=tuple(pre),
        post=tuple(post),
        insert_with_body=insert_body,
        params=_get_param_names(fn),
    )


def _extract_called_name(call: PyIRCall) -> str | None:
    if isinstance(call.func, PyIRVarUse):
        return call.func.name
    return None


def _clone_with_subst(node: PyIRNode, subst: Dict[str, PyIRNode]) -> PyIRNode:
    # По твоему инварианту вложенных def внутри шаблона быть не должно.
    # Но как предохранитель — не трогаем их, если вдруг утекут.
    if isinstance(node, PyIRFunctionDef):
        return deepcopy(node)

    if isinstance(node, PyIRVarUse) and node.name in subst:
        return deepcopy(subst[node.name])

    if not is_dataclass(node):
        return deepcopy(node)

    clone = deepcopy(node)
    for f in fields(clone):
        v = getattr(clone, f.name)

        if isinstance(v, PyIRNode):
            setattr(clone, f.name, _clone_with_subst(v, subst))
            continue

        if isinstance(v, list):
            new_list: list[object] = []
            for item in v:
                if isinstance(item, PyIRNode):
                    new_list.append(_clone_with_subst(item, subst))
                else:
                    new_list.append(item)
            setattr(clone, f.name, new_list)
            continue

        if isinstance(v, dict):
            new_dict: dict[object, object] = {}
            for key, item in v.items():
                new_key = (
                    _clone_with_subst(key, subst) if isinstance(key, PyIRNode) else key
                )
                new_item = (
                    _clone_with_subst(item, subst)
                    if isinstance(item, PyIRNode)
                    else item
                )
                new_dict[new_key] = new_item
            setattr(clone, f.name, new_dict)

    return clone


def _instantiate_template(
    tmpl: ContextManagerTemplate, call: PyIRCall
) -> Tuple[List[PyIRNode], List[PyIRNode]]:
    # Здесь НЕТ никакого "если arg is call -> tmp".
    # Это обязан сделать NormalizeCallArgumentsPass ДО with.
    subst: Dict[str, PyIRNode] = {}

    # positional first
    for i, p in enumerate(tmpl.params):
        if i < len(call.args_p):
            subst[p] = deepcopy(call.args_p[i])

    # keyword (на всякий, но после нормализации ожидается пусто)
    for k, v in call.args_kw.items():
        subst[k] = deepcopy(v)

    pre = [_clone_with_subst(st, subst) for st in tmpl.pre]
    post = [_clone_with_subst(st, subst) for st in tmpl.post]
    return pre, post


class CollectContextManagersPass:
    """
    Expand pass #1:
    - собирает contextmanager-функции в шаблоны
    - дубликаты имён: последняя дефиниция побеждает (как Python)
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> None:
        for node in ir.walk():
            if not isinstance(node, PyIRFunctionDef):
                continue

            if any(
                _compiler_directive_kind(d) == "contextmanager" for d in node.decorators
            ):
                ctx.cm_templates[node.name] = _split_contextmanager_fn(ir, node)


class RewriteWithContextManagerPass:
    """
    Expand pass #2:
    - with a(...), b(...): стек (enter L->R, exit R->L)
    - если item не наш (не call / не contextmanager) -> не трогаем with
    - 0 marker -> with-body не вставляется, и последующие items не выполняются
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        ir.body = RewriteWithContextManagerPass._rewrite_stmt_list(ir, ir.body, ctx)
        return ir

    @staticmethod
    def _rewrite_stmt_list(
        ir: PyIRFile, stmts: List[PyIRNode], ctx: ExpandContext
    ) -> List[PyIRNode]:
        out: List[PyIRNode] = []

        for st in stmts:
            RewriteWithContextManagerPass._rewrite_children_blocks(ir, st, ctx)

            if isinstance(st, PyIRWith):
                out.extend(RewriteWithContextManagerPass._expand_with(ir, st, ctx))
            else:
                out.append(st)

        return out

    @staticmethod
    def _rewrite_children_blocks(
        ir: PyIRFile, node: PyIRNode, ctx: ExpandContext
    ) -> None:
        if not is_dataclass(node):
            return

        for f in fields(node):
            val = getattr(node, f.name)
            if (
                isinstance(val, list)
                and val
                and all(isinstance(x, PyIRNode) for x in val)
            ):
                setattr(
                    node,
                    f.name,
                    RewriteWithContextManagerPass._rewrite_stmt_list(ir, val, ctx),
                )

    @staticmethod
    def _expand_with(
        ir: PyIRFile, node: PyIRWith, ctx: ExpandContext
    ) -> List[PyIRNode]:
        if not node.items:
            return [node]

        prepared: List[Tuple[ContextManagerTemplate, PyIRCall]] = []

        for it in node.items:
            expr = it.context_expr
            if not isinstance(expr, PyIRCall):
                return [node]

            fn_name = _extract_called_name(expr)
            if fn_name is None:
                return [node]

            tmpl = ctx.cm_templates.get(fn_name)
            if tmpl is None:
                return [node]

            prepared.append((tmpl, expr))

        prefix: List[PyIRNode] = []
        post_stack: List[List[PyIRNode]] = []
        body: List[PyIRNode] = node.body

        for tmpl, call in prepared:
            pre, post = _instantiate_template(tmpl, call)
            prefix.extend(pre)
            post_stack.append(post)

            if not tmpl.insert_with_body:
                body = []
                break

        expanded: List[PyIRNode] = []
        expanded.extend(prefix)
        expanded.extend(body)

        for post in reversed(post_stack):
            expanded.extend(post)

        return expanded

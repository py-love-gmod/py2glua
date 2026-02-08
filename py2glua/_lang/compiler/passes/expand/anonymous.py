from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Dict, List

from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRCall,
    PyIRDecorator,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRReturn,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
)
from ...compiler_ir import PyIRFunctionExpr
from .expand_context import ExpandContext

_CD_PREFIX = (
    "py2glua",
    "glua",
    "core",
    "compiler_directive",
    "CompilerDirective",
)


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

    return tuple(parts) == _CD_PREFIX


def _compiler_directive_kind(dec: PyIRDecorator) -> str | None:
    expr = dec.exper

    # @CompilerDirective.anonymous
    if isinstance(expr, PyIRAttribute):
        if _is_compiler_directive_attr(expr, method="anonymous"):
            return "anonymous"
        return None

    # @CompilerDirective.anonymous()
    if isinstance(expr, PyIRCall):
        f = expr.func
        if isinstance(f, PyIRAttribute):
            if _is_compiler_directive_attr(f, method="anonymous"):
                return "anonymous"
        return None

    return None


def _is_anonymous_def(fn: PyIRFunctionDef) -> bool:
    return any(_compiler_directive_kind(d) == "anonymous" for d in fn.decorators)


class CollectAnonymousFunctionsPass:
    """
    Собирает шаблоны anonymous-функций:

        @CompilerDirective.anonymous()
        def foo(...): ...

    Сохраняет в ctx.anonymous_templates:
        name -> PyIRFunctionExpr(signature, body)

    Ничего не переписывает.
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> None:
        for node in ir.walk():
            if not isinstance(node, PyIRFunctionDef):
                continue

            if not _is_anonymous_def(node):
                continue

            ctx.anonymous_templates[node.name] = PyIRFunctionExpr(
                line=node.line,
                offset=node.offset,
                signature=deepcopy(node.signature),
                vararg=node.vararg,
                kwarg=node.kwarg,
                body=deepcopy(node.body),
            )


class RewriteAnonymousFunctionsPass:
    """
    1) Удаляет сами def foo, помеченные anonymous (они больше не нужны в emit)
    2) Заменяет VarUse(foo) -> PyIRFunctionExpr (function-literal)
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        ir.body = RewriteAnonymousFunctionsPass._rw_stmt_list(ir, ir.body, ctx)
        return ir

    @staticmethod
    def _rw_stmt_list(
        ir: PyIRFile, stmts: List[PyIRNode], ctx: ExpandContext
    ) -> List[PyIRNode]:
        out: List[PyIRNode] = []
        for st in stmts:
            out.extend(RewriteAnonymousFunctionsPass._rw_stmt(ir, st, ctx))
        return out

    @staticmethod
    def _rw_stmt(ir: PyIRFile, st: PyIRNode, ctx: ExpandContext) -> List[PyIRNode]:
        if isinstance(st, PyIRFunctionDef) and _is_anonymous_def(st):
            return []

        if isinstance(st, PyIRIf):
            pro, test = RewriteAnonymousFunctionsPass._rw_expr(ir, st.test, ctx)
            st.test = test
            st.body = RewriteAnonymousFunctionsPass._rw_stmt_list(ir, st.body, ctx)
            st.orelse = RewriteAnonymousFunctionsPass._rw_stmt_list(ir, st.orelse, ctx)
            return pro + [st]

        if isinstance(st, PyIRWhile):
            pro, test = RewriteAnonymousFunctionsPass._rw_expr(ir, st.test, ctx)
            st.test = test
            st.body = RewriteAnonymousFunctionsPass._rw_stmt_list(ir, st.body, ctx)
            return pro + [st]

        if isinstance(st, PyIRFor):
            pro_iter, it = RewriteAnonymousFunctionsPass._rw_expr(ir, st.iter, ctx)
            st.iter = it
            st.body = RewriteAnonymousFunctionsPass._rw_stmt_list(ir, st.body, ctx)
            return pro_iter + [st]

        if isinstance(st, PyIRWith):
            pro_with: List[PyIRNode] = []
            for item in st.items:
                p, e = RewriteAnonymousFunctionsPass._rw_expr(
                    ir, item.context_expr, ctx
                )
                pro_with.extend(p)
                item.context_expr = e
            st.body = RewriteAnonymousFunctionsPass._rw_stmt_list(ir, st.body, ctx)
            return pro_with + [st]

        if isinstance(st, PyIRFunctionDef):
            st.body = RewriteAnonymousFunctionsPass._rw_stmt_list(ir, st.body, ctx)
            return [st]

        if isinstance(st, PyIRAssign):
            pro, v = RewriteAnonymousFunctionsPass._rw_expr(ir, st.value, ctx)
            st.value = v
            return pro + [st]

        if isinstance(st, PyIRReturn):
            if st.value is None:
                return [st]

            pro_ret, value = RewriteAnonymousFunctionsPass._rw_expr(ir, st.value, ctx)
            st.value = value
            return pro_ret + [st]

        if isinstance(st, PyIRCall):
            pro, c = RewriteAnonymousFunctionsPass._rw_expr(ir, st, ctx)
            return pro + [c]

        return [st]

    @staticmethod
    def _rw_expr(
        ir: PyIRFile, expr: PyIRNode, ctx: ExpandContext
    ) -> tuple[List[PyIRNode], PyIRNode]:
        if isinstance(expr, PyIRCall):
            pro: List[PyIRNode] = []

            p_func, nf = RewriteAnonymousFunctionsPass._rw_expr(ir, expr.func, ctx)
            pro.extend(p_func)
            expr.func = nf

            new_args_p: List[PyIRNode] = []
            for a in expr.args_p:
                p, na = RewriteAnonymousFunctionsPass._rw_expr(ir, a, ctx)
                pro.extend(p)
                new_args_p.append(na)
            expr.args_p = new_args_p

            new_args_kw: Dict[str, PyIRNode] = {}
            for k, a in expr.args_kw.items():
                p, na = RewriteAnonymousFunctionsPass._rw_expr(ir, a, ctx)
                pro.extend(p)
                new_args_kw[k] = na
            expr.args_kw = new_args_kw

            return pro, expr

        if isinstance(expr, PyIRVarUse):
            tmpl = ctx.anonymous_templates.get(expr.name)
            if tmpl is None:
                return [], expr

            fnexpr = deepcopy(tmpl)
            fnexpr.line = expr.line
            fnexpr.offset = expr.offset
            return [], fnexpr

        if not is_dataclass(expr):
            return [], expr

        pro_other: List[PyIRNode] = []

        for f in fields(expr):
            v = getattr(expr, f.name)

            if isinstance(v, PyIRNode):
                p, nv = RewriteAnonymousFunctionsPass._rw_expr(ir, v, ctx)
                pro_other.extend(p)
                setattr(expr, f.name, nv)
                continue

            if isinstance(v, list):
                new_list: List[object] = []
                changed = False
                for x in v:
                    if isinstance(x, PyIRNode):
                        p, nx = RewriteAnonymousFunctionsPass._rw_expr(ir, x, ctx)
                        pro_other.extend(p)
                        new_list.append(nx)
                        changed = True
                    else:
                        new_list.append(x)
                if changed:
                    setattr(expr, f.name, new_list)
                continue

            if isinstance(v, dict):
                new_dict: Dict[str, object] = {}
                changed = False
                for k, x in v.items():
                    if isinstance(x, PyIRNode):
                        p, nx = RewriteAnonymousFunctionsPass._rw_expr(ir, x, ctx)
                        pro_other.extend(p)
                        new_dict[k] = nx
                        changed = True
                    else:
                        new_dict[k] = x
                if changed:
                    setattr(expr, f.name, new_dict)
                continue

        return pro_other, expr

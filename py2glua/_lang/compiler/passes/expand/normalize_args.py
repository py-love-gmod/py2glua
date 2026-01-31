from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRCall,
    PyIRClassDef,
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
from .expand_context import ExpandContext, FnSig


def _mk_var(name: str, ref: PyIRNode) -> PyIRVarUse:
    return PyIRVarUse(line=ref.line, offset=ref.offset, name=name)


def _mk_assign(name: str, value: PyIRNode, ref: PyIRNode) -> PyIRAssign:
    return PyIRAssign(
        line=ref.line,
        offset=ref.offset,
        targets=[_mk_var(name, ref)],
        value=value,
    )


class CollectLocalSignaturesPass:
    """
    Собирает сигнатуры top-level функций текущего файла в ctx.fn_sigs.
    Последняя дефиниция побеждает (как в Python).
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> None:
        for st in ir.body:
            if not isinstance(st, PyIRFunctionDef):
                continue

            params = tuple(st.signature.keys())
            defaults: Dict[str, PyIRNode] = {}

            for name, (_ann, default) in st.signature.items():
                if default is not None:
                    defaults[name] = default

            ctx.fn_sigs[st.name] = FnSig(params=params, defaults=defaults)


class NormalizeCallArgumentsPass:
    """
    - Hoist nested PyIRCall used as an argument of another PyIRCall into temp vars.
    - Convert keyword args -> positional args for calls to locally-known functions
      (ctx.fn_sigs), using defaults when needed.
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        sigs = ctx.fn_sigs
        ir.body = NormalizeCallArgumentsPass._norm_stmt_list(ir, ir.body, ctx, sigs)
        return ir

    @staticmethod
    def _norm_stmt_list(
        ir: PyIRFile,
        stmts: List[PyIRNode],
        ctx: ExpandContext,
        sigs: Dict[str, FnSig],
    ) -> List[PyIRNode]:
        out: List[PyIRNode] = []
        for st in stmts:
            out.extend(NormalizeCallArgumentsPass._norm_stmt(ir, st, ctx, sigs))

        return out

    @staticmethod
    def _norm_stmt(
        ir: PyIRFile,
        st: PyIRNode,
        ctx: ExpandContext,
        sigs: Dict[str, FnSig],
    ) -> List[PyIRNode]:
        if isinstance(st, PyIRIf):
            pro, test = NormalizeCallArgumentsPass._norm_expr(
                ir, st.test, ctx, sigs, in_call_arg=False
            )
            st.test = test
            st.body = NormalizeCallArgumentsPass._norm_stmt_list(ir, st.body, ctx, sigs)
            st.orelse = NormalizeCallArgumentsPass._norm_stmt_list(
                ir, st.orelse, ctx, sigs
            )
            return pro + [st]

        if isinstance(st, PyIRWhile):
            pro, test = NormalizeCallArgumentsPass._norm_expr(
                ir, st.test, ctx, sigs, in_call_arg=False
            )
            st.test = test
            st.body = NormalizeCallArgumentsPass._norm_stmt_list(ir, st.body, ctx, sigs)
            return pro + [st]

        if isinstance(st, PyIRFor):
            pro, it = NormalizeCallArgumentsPass._norm_expr(
                ir, st.iter, ctx, sigs, in_call_arg=False
            )
            st.iter = it
            st.body = NormalizeCallArgumentsPass._norm_stmt_list(ir, st.body, ctx, sigs)
            return pro + [st]

        if isinstance(st, PyIRWith):
            pro: List[PyIRNode] = []
            for item in st.items:
                p, e = NormalizeCallArgumentsPass._norm_expr(
                    ir, item.context_expr, ctx, sigs, in_call_arg=False
                )
                pro.extend(p)
                item.context_expr = e

            st.body = NormalizeCallArgumentsPass._norm_stmt_list(ir, st.body, ctx, sigs)
            return pro + [st]

        if isinstance(st, PyIRFunctionDef):
            st.body = NormalizeCallArgumentsPass._norm_stmt_list(ir, st.body, ctx, sigs)
            return [st]

        if isinstance(st, PyIRClassDef):
            st.body = NormalizeCallArgumentsPass._norm_stmt_list(ir, st.body, ctx, sigs)
            return [st]

        if isinstance(st, PyIRAssign):
            pro, v = NormalizeCallArgumentsPass._norm_expr(
                ir, st.value, ctx, sigs, in_call_arg=False
            )
            st.value = v
            return pro + [st]

        if isinstance(st, PyIRReturn):
            pro, v = NormalizeCallArgumentsPass._norm_expr(
                ir, st.value, ctx, sigs, in_call_arg=False
            )
            st.value = v
            return pro + [st]

        if isinstance(st, PyIRCall):
            pro, c = NormalizeCallArgumentsPass._norm_expr(
                ir, st, ctx, sigs, in_call_arg=False
            )
            return pro + [c]

        return [st]

    @staticmethod
    def _norm_expr(
        ir: PyIRFile,
        expr: PyIRNode,
        ctx: ExpandContext,
        sigs: Dict[str, FnSig],
        *,
        in_call_arg: bool,
    ) -> Tuple[List[PyIRNode], PyIRNode]:
        if isinstance(expr, PyIRCall):
            pro: List[PyIRNode] = []

            p_func, new_func = NormalizeCallArgumentsPass._norm_expr(
                ir, expr.func, ctx, sigs, in_call_arg=False
            )
            pro.extend(p_func)

            new_args_p: List[PyIRNode] = []
            for a in expr.args_p:
                p, na = NormalizeCallArgumentsPass._norm_expr(
                    ir, a, ctx, sigs, in_call_arg=True
                )
                pro.extend(p)
                new_args_p.append(na)

            new_args_kw: Dict[str, PyIRNode] = {}
            for k, a in expr.args_kw.items():
                p, na = NormalizeCallArgumentsPass._norm_expr(
                    ir, a, ctx, sigs, in_call_arg=True
                )
                pro.extend(p)
                new_args_kw[k] = na

            new_call = PyIRCall(
                line=expr.line,
                offset=expr.offset,
                func=new_func,
                args_p=new_args_p,
                args_kw=new_args_kw,
            )

            new_call = NormalizeCallArgumentsPass._kw_to_pos_if_possible(
                ir, new_call, sigs
            )

            if in_call_arg:
                tmp = ctx.new_tmp_name()
                pro.append(_mk_assign(tmp, new_call, expr))
                return pro, _mk_var(tmp, expr)

            return pro, new_call

        return [], expr

    @staticmethod
    def _kw_to_pos_if_possible(
        ir: PyIRFile, call: PyIRCall, sigs: Dict[str, FnSig]
    ) -> PyIRCall:
        if not isinstance(call.func, PyIRVarUse):
            return call

        fn_name = call.func.name
        sig = sigs.get(fn_name)
        if sig is None:
            return call

        params = sig.params
        defaults = sig.defaults

        provided: Dict[str, PyIRNode] = {}

        extra_pos: List[PyIRNode] = []
        for i, a in enumerate(call.args_p):
            if i < len(params):
                provided[params[i]] = a
            else:
                extra_pos.append(a)

        for k, v in call.args_kw.items():
            if k not in params:
                CompilerExit.user_error_node(
                    f"Неизвестный keyword-аргумент '{k}' для вызова '{fn_name}'",
                    ir.path,
                    call,
                )

            if k in provided:
                CompilerExit.user_error_node(
                    f"Аргумент '{k}' для '{fn_name}' передан дважды (positional + keyword)",
                    ir.path,
                    call,
                )

            provided[k] = v

        new_args_p: List[PyIRNode] = []
        for p in params:
            if p in provided:
                new_args_p.append(provided[p])
                continue

            d = defaults.get(p)
            if d is None:
                CompilerExit.user_error_node(
                    f"Не хватает обязательного аргумента '{p}' для вызова '{fn_name}'",
                    ir.path,
                    call,
                )

            new_args_p.append(deepcopy(d))

        new_args_p.extend(extra_pos)

        return PyIRCall(
            line=call.line,
            offset=call.offset,
            func=call.func,
            args_p=new_args_p,
            args_kw={},
        )

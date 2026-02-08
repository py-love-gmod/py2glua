from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRDict,
    PyIRDictItem,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRList,
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
    Собирает сигнатуры top-level функций текущего файла в `ctx.fn_sigs`.
    Последняя дефиниция побеждает (как в Python).
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> None:
        file_key = CollectLocalSignaturesPass._file_key(ir.path)
        file_sigs: Dict[str, FnSig] = {}

        for st in ir.body:
            if not isinstance(st, PyIRFunctionDef):
                continue

            params = tuple(st.signature.keys())
            defaults: Dict[str, PyIRNode] = {}

            for name, (_ann, default) in st.signature.items():
                if default is not None:
                    defaults[name] = default

            file_sigs[st.name] = FnSig(
                params=params,
                defaults=defaults,
                vararg_name=st.vararg,
                kwarg_name=st.kwarg,
            )

        ctx.fn_sigs[file_key] = file_sigs

    @staticmethod
    def _file_key(path: Path | None) -> str:
        if path is None:
            return "<none>"
        try:
            return str(path.resolve())
        except Exception:
            return str(path)


class NormalizeCallArgumentsPass:
    """
    - Поднимает вложенные `PyIRCall`, используемые как аргументы вызова, во временные переменные.
    - Нормализует вызовы локально известных функций:
      positional + keyword -> только positional, с подстановкой default-значений.
    - Для `*args` упаковывает лишние positional в `PyIRList`.
    - Для `**kwargs` упаковывает лишние keyword в `PyIRDict`.
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        sigs = ctx.fn_sigs.get(CollectLocalSignaturesPass._file_key(ir.path), {})
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
            pro_with: List[PyIRNode] = []
            for item in st.items:
                p, e = NormalizeCallArgumentsPass._norm_expr(
                    ir, item.context_expr, ctx, sigs, in_call_arg=False
                )
                pro_with.extend(p)
                item.context_expr = e

            st.body = NormalizeCallArgumentsPass._norm_stmt_list(ir, st.body, ctx, sigs)
            return pro_with + [st]

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
            if st.value is None:
                return [st]

            pro_ret, value = NormalizeCallArgumentsPass._norm_expr(
                ir, st.value, ctx, sigs, in_call_arg=False
            )
            st.value = value
            return pro_ret + [st]

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
                        f"Аргумент '{key}' для '{fn_name}' передан дважды (позиционно и по имени)",
                        ir.path,
                        call,
                    )
                provided[key] = value
                continue

            extra_kw[key] = value

        if extra_pos and sig.vararg_name is None:
            CompilerExit.user_error_node(
                f"Слишком много позиционных аргументов при вызове '{fn_name}'",
                ir.path,
                call,
            )

        if extra_kw and sig.kwarg_name is None:
            bad_names = ", ".join(sorted(extra_kw.keys()))
            CompilerExit.user_error_node(
                f"Неизвестные keyword-аргументы для вызова '{fn_name}': {bad_names}",
                ir.path,
                call,
            )

        new_args_p: List[PyIRNode] = []
        for param_name in params:
            if param_name in provided:
                new_args_p.append(provided[param_name])
                continue

            default_value = defaults.get(param_name)
            if default_value is None:
                CompilerExit.user_error_node(
                    f"Не хватает обязательного аргумента '{param_name}' для вызова '{fn_name}'",
                    ir.path,
                    call,
                )

            new_args_p.append(deepcopy(default_value))

        if sig.vararg_name is not None:
            new_args_p.append(
                PyIRList(
                    line=call.line,
                    offset=call.offset,
                    elements=extra_pos,
                )
            )

        if sig.kwarg_name is not None:
            kw_items: List[PyIRDictItem] = [
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

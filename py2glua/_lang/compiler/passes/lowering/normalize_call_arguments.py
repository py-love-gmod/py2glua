from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRCall,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
)
from ..analysis.symlinks import PyIRSymLink, SymbolId, SymLinkContext
from ..clone_utils import clone_with_pos
from ..macro_utils import MacroUtils


@dataclass(frozen=True)
class _FnSigInfo:
    sym_id: int
    name: str
    file: Path | None
    line: int | None
    offset: int | None
    param_names: Tuple[str, ...]
    param_index: Dict[str, int]
    default_expr: Tuple[PyIRNode | None, ...]


@dataclass
class _CallJob:
    call: PyIRCall
    file: Path | None


class NormalizeCallArgumentsPass:
    """
    Нормализует вызовы функций:
      - переносит kwargs в позиционные аргументы
      - подставляет значения по умолчанию

    Инварианты после пасса:
      - call.args_kw == {}
      - len(call.args_p) == len(signature)
    """

    _CTX_SEEN = "_callnorm_seen_paths"
    _CTX_DONE = "_callnorm_done"
    _CTX_FUNCS = "_callnorm_funcs"  # sym_id -> _FnSigInfo
    _CTX_CALLS = "_callnorm_calls"  # list[_CallJob]

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        MacroUtils.ensure_pass_ctx(
            ctx,
            **{
                NormalizeCallArgumentsPass._CTX_SEEN: set(),
                NormalizeCallArgumentsPass._CTX_DONE: False,
                NormalizeCallArgumentsPass._CTX_FUNCS: {},
                NormalizeCallArgumentsPass._CTX_CALLS: [],
            },
        )

        if ir.path is not None:
            getattr(ctx, NormalizeCallArgumentsPass._CTX_SEEN).add(ir.path.resolve())

        NormalizeCallArgumentsPass._collect_functions(ir, ctx)
        NormalizeCallArgumentsPass._collect_calls(ir, ctx)

        if MacroUtils.ready_to_finalize(
            ctx,
            seen_key=NormalizeCallArgumentsPass._CTX_SEEN,
            done_key=NormalizeCallArgumentsPass._CTX_DONE,
        ):
            funcs: Dict[int, _FnSigInfo] = getattr(
                ctx, NormalizeCallArgumentsPass._CTX_FUNCS
            )
            calls: list[_CallJob] = getattr(ctx, NormalizeCallArgumentsPass._CTX_CALLS)
            for job in calls:
                NormalizeCallArgumentsPass._normalize_call(job.call, job.file, funcs)

    @staticmethod
    def _func_symbol_id(fn: PyIRFunctionDef, ctx: SymLinkContext) -> int | None:
        decl_scope = ctx.node_scope[ctx.uid(fn)]
        link = ctx.resolve(scope=decl_scope, name=fn.name)
        return None if link is None else link.target.value

    @staticmethod
    def _collect_functions(ir: PyIRFile, ctx: SymLinkContext) -> None:
        funcs: Dict[int, _FnSigInfo] = getattr(
            ctx, NormalizeCallArgumentsPass._CTX_FUNCS
        )

        for node in ir.walk():
            if not isinstance(node, PyIRFunctionDef):
                continue

            sym_id = NormalizeCallArgumentsPass._func_symbol_id(node, ctx)
            if sym_id is None or sym_id in funcs:
                continue

            sym = ctx.symbols.get(SymbolId(sym_id))

            items = list(node.signature.items())
            param_names = tuple(k for k, _ in items)
            default_expr = tuple(v[1] for _, v in items)
            param_index = {name: i for i, name in enumerate(param_names)}

            funcs[sym_id] = _FnSigInfo(
                sym_id=sym_id,
                name=node.name,
                file=sym.decl_file if sym else ir.path,
                line=sym.decl_line if sym else node.line,
                offset=node.offset,
                param_names=param_names,
                param_index=param_index,
                default_expr=default_expr,
            )

    @staticmethod
    def _collect_calls(ir: PyIRFile, ctx: SymLinkContext) -> None:
        calls: list[_CallJob] = getattr(ctx, NormalizeCallArgumentsPass._CTX_CALLS)
        for node in ir.walk():
            if isinstance(node, PyIRCall):
                calls.append(_CallJob(call=node, file=ir.path))

    @staticmethod
    def _normalize_call(
        call: PyIRCall,
        file: Path | None,
        funcs: Dict[int, _FnSigInfo],
    ) -> None:
        if not isinstance(call.func, PyIRSymLink):
            return

        fn = funcs.get(call.func.symbol_id)
        if fn is None:
            return

        param_names = fn.param_names
        nparams = len(param_names)

        if len(call.args_p) > nparams:
            CompilerExit.user_error_node(
                "Слишком много позиционных аргументов при вызове функции.\n"
                f"Функция: {fn.name}\n"
                f"Ожидается: {nparams}, получено: {len(call.args_p)}",
                file,
                call,
            )
            assert False

        slots: list[PyIRNode | None] = [None] * nparams

        for i, arg in enumerate(call.args_p):
            slots[i] = arg

        for k, v in call.args_kw.items():
            idx = fn.param_index.get(k)
            if idx is None:
                CompilerExit.user_error_node(
                    "Передан неизвестный именованный аргумент при вызове функции.\n"
                    f"Функция: {fn.name}\n"
                    f"Аргумент: {k}",
                    file,
                    call,
                )
                assert False

            if slots[idx] is not None:
                CompilerExit.user_error_node(
                    "Параметр передан дважды (позиционно и/или по имени).\n"
                    f"Функция: {fn.name}\n"
                    f"Параметр: {k}",
                    file,
                    call,
                )
                assert False

            slots[idx] = v

        for i in range(nparams):
            if slots[i] is not None:
                continue

            d = fn.default_expr[i]
            if d is None:
                CompilerExit.user_error_node(
                    "Не передан обязательный аргумент.\n"
                    f"Функция: {fn.name}\n"
                    f"Параметр: {param_names[i]}",
                    file,
                    call,
                )
                assert False

            slots[i] = clone_with_pos(
                d,
                line=call.line,
                offset=call.offset,
            )

        call.args_p = slots  # type: ignore[assignment]
        call.args_kw = {}

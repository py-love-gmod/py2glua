from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from ....._cli.logging_setup import exit_with_code
from ....py.ir_dataclass import (
    PyIRCall,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
)
from ..analysis.symlinks import PyIRSymLink, SymbolId, SymLinkContext


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
        NormalizeCallArgumentsPass._ensure_ctx(ctx)

        if ir.path is not None:
            getattr(ctx, NormalizeCallArgumentsPass._CTX_SEEN).add(ir.path.resolve())

        NormalizeCallArgumentsPass._collect_functions(ir, ctx)
        NormalizeCallArgumentsPass._collect_calls(ir, ctx)
        NormalizeCallArgumentsPass._maybe_finalize(ctx)

    @staticmethod
    def _ensure_ctx(ctx: SymLinkContext) -> None:
        if not hasattr(ctx, NormalizeCallArgumentsPass._CTX_SEEN):
            setattr(ctx, NormalizeCallArgumentsPass._CTX_SEEN, set())

        if not hasattr(ctx, NormalizeCallArgumentsPass._CTX_DONE):
            setattr(ctx, NormalizeCallArgumentsPass._CTX_DONE, False)

        if not hasattr(ctx, NormalizeCallArgumentsPass._CTX_FUNCS):
            setattr(ctx, NormalizeCallArgumentsPass._CTX_FUNCS, {})

        if not hasattr(ctx, NormalizeCallArgumentsPass._CTX_CALLS):
            setattr(ctx, NormalizeCallArgumentsPass._CTX_CALLS, [])

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
    def _maybe_finalize(ctx: SymLinkContext) -> None:
        if getattr(ctx, NormalizeCallArgumentsPass._CTX_DONE):
            return

        all_paths = {p.resolve() for p in getattr(ctx, "_all_paths", set()) if p}
        seen = getattr(ctx, NormalizeCallArgumentsPass._CTX_SEEN)
        if all_paths and seen != all_paths:
            return

        setattr(ctx, NormalizeCallArgumentsPass._CTX_DONE, True)

        funcs: Dict[int, _FnSigInfo] = getattr(
            ctx, NormalizeCallArgumentsPass._CTX_FUNCS
        )
        calls: list[_CallJob] = getattr(ctx, NormalizeCallArgumentsPass._CTX_CALLS)

        for job in calls:
            NormalizeCallArgumentsPass._normalize_call(job.call, job.file, funcs)

    @staticmethod
    def _err_ctx(file: Path | None, line: int | None, offset: int | None) -> str:
        return f"Файл: {file}\nLINE|OFFSET: {line}|{offset}"

    @staticmethod
    def _clone_node_set_pos(
        node: PyIRNode, *, line: int | None, offset: int | None
    ) -> PyIRNode:
        from dataclasses import fields, is_dataclass

        def clone_value(v):
            if isinstance(v, PyIRNode):
                return clone_node(v)

            if isinstance(v, list):
                return [clone_value(x) for x in v]

            if isinstance(v, dict):
                return {k: clone_value(x) for k, x in v.items()}

            if isinstance(v, tuple):
                return tuple(clone_value(x) for x in v)

            return v

        def clone_node(n: PyIRNode) -> PyIRNode:
            if not is_dataclass(n):
                return n

            cls = n.__class__
            kwargs = {}
            for f in fields(n):
                if f.name == "line":
                    kwargs[f.name] = line

                elif f.name == "offset":
                    kwargs[f.name] = offset

                else:
                    kwargs[f.name] = clone_value(getattr(n, f.name))

            return cls(**kwargs)

        return clone_node(node)

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
            exit_with_code(
                1,
                "Слишком много позиционных аргументов при вызове функции.\n"
                f"Функция: {fn.name}\n"
                f"Ожидается: {nparams}, получено: {len(call.args_p)}\n"
                + NormalizeCallArgumentsPass._err_ctx(file, call.line, call.offset),
            )
            assert False

        slots: list[PyIRNode | None] = [None] * nparams

        for i, arg in enumerate(call.args_p):
            slots[i] = arg

        for k, v in call.args_kw.items():
            idx = fn.param_index.get(k)
            if idx is None:
                exit_with_code(
                    1,
                    "Передан неизвестный именованный аргумент при вызове функции.\n"
                    f"Функция: {fn.name}\n"
                    f"Аргумент: {k}\n"
                    + NormalizeCallArgumentsPass._err_ctx(file, call.line, call.offset),
                )
                assert False

            if slots[idx] is not None:
                exit_with_code(
                    1,
                    "Параметр передан дважды (позиционно и/или по имени).\n"
                    f"Функция: {fn.name}\n"
                    f"Параметр: {k}\n"
                    + NormalizeCallArgumentsPass._err_ctx(file, call.line, call.offset),
                )
                assert False

            slots[idx] = v

        for i in range(nparams):
            if slots[i] is not None:
                continue

            d = fn.default_expr[i]
            if d is None:
                exit_with_code(
                    1,
                    "Не передан обязательный аргумент.\n"
                    f"Функция: {fn.name}\n"
                    f"Параметр: {param_names[i]}\n"
                    + NormalizeCallArgumentsPass._err_ctx(file, call.line, call.offset),
                )
                assert False

            slots[i] = NormalizeCallArgumentsPass._clone_node_set_pos(
                d,
                line=call.line,
                offset=call.offset,
            )

        call.args_p = slots  # type: ignore
        call.args_kw = {}

from __future__ import annotations

from typing import Iterable

from ....._cli.logging_setup import exit_with_code
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import PyIRAttribute, PyIRFunctionDef, PyIRNode, PyIRVarUse
from ...ir_compiler import PyIRSymbolRef
from .ctx import AnalysisContext, AnalysisPass, ContextManagerInfo, SymbolInfo


class CollectContextManagersPass(AnalysisPass):
    TARGET_DECORATOR = "CompilerDirective.contextmanager"
    MARKER_ATTR = "contextmanager_body"
    MARKER_OWNER_NAME = "CompilerDirective"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if ir.path is None:
            return

        for fn in cls._iter_module_functions(ir):
            if not cls._has_contextmanager_decorator(fn):
                continue

            sid = cls._get_function_symbol_id(fn.name, ir, ctx)
            if sid is None:
                continue

            pre, post = cls._split_by_marker(fn.body, ctx, ir)

            info = ContextManagerInfo(
                function_symbol_id=sid,
                fqname=cls._get_fqname(sid, ctx),
                file=ir.path,
                line=getattr(fn, "line", None),
                offset=getattr(fn, "offset", None),
                pre_body=pre,
                post_body=post,
            )

            ctx.context_managers[sid] = info

    @classmethod
    def _split_by_marker(
        cls,
        body: list[PyIRNode],
        ctx: AnalysisContext,
        ir: PyIRFile,
    ) -> tuple[list[PyIRNode], list[PyIRNode]]:
        marker_idx: int | None = None

        for i, stmt in enumerate(body):
            if not cls._is_marker_stmt(stmt, ctx):
                continue

            if marker_idx is not None:
                cls._error(
                    ir=ir,
                    line=getattr(stmt, "line", None),
                    offset=getattr(stmt, "offset", None),
                    msg=(
                        "В contextmanager-функции найдено более одного маркера "
                        "CompilerDirective.contextmanager_body."
                    ),
                )
                assert False, "unreachable"

            marker_idx = i

        if marker_idx is None:
            return list(body), []

        pre = list(body[:marker_idx])
        post = list(body[marker_idx + 1 :])
        return pre, post

    @classmethod
    def _is_marker_stmt(cls, node: PyIRNode, ctx: AnalysisContext) -> bool:
        if isinstance(node, PyIRAttribute):
            return cls._is_marker_expr(node, ctx)

        val = getattr(node, "value", None)
        if isinstance(val, PyIRAttribute):
            return cls._is_marker_expr(val, ctx)

        return False

    @classmethod
    def _is_marker_expr(cls, expr: PyIRAttribute, ctx: AnalysisContext) -> bool:
        if expr.attr != cls.MARKER_ATTR:
            return False

        base = expr.value

        if isinstance(base, PyIRVarUse):
            return base.name == cls.MARKER_OWNER_NAME

        if isinstance(base, PyIRSymbolRef):
            info = ctx.symbols.get(base.sym_id)
            if info is None:
                return False

            return info.symbol.name == cls.MARKER_OWNER_NAME

        return False

    @staticmethod
    def _iter_module_functions(ir: PyIRFile) -> Iterable[PyIRFunctionDef]:
        for node in ir.body:
            if isinstance(node, PyIRFunctionDef):
                yield node

    @classmethod
    def _has_contextmanager_decorator(cls, fn: PyIRFunctionDef) -> bool:
        if not fn.decorators:
            return False
        return any(d.name == cls.TARGET_DECORATOR for d in fn.decorators)

    @staticmethod
    def _get_function_symbol_id(
        name: str,
        ir: PyIRFile,
        ctx: AnalysisContext,
    ) -> int | None:
        if ir.path is None:
            return None

        info: SymbolInfo
        for sid, info in ctx.symbols.items():
            if info.file != ir.path:
                continue

            sym = info.symbol
            if sym.scope == "module" and sym.kind == "function" and sym.name == name:
                return sid

        return None

    @staticmethod
    def _get_fqname(sid: int, ctx: AnalysisContext) -> str:
        info = ctx.symbols.get(sid)
        if info is None:
            return f"<sym:{sid}>"

        return info.fqname

    @staticmethod
    def _error(*, ir: PyIRFile, line: int | None, offset: int | None, msg: str) -> None:
        full_msg = f"{msg}\nФайл: {ir.path}\nLINE|OFFSET: {line}|{offset}"
        exit_with_code(1, full_msg)
        assert False, "unreachable"

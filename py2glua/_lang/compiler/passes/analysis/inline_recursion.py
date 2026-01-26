from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Iterable

from ....._cli.logging_setup import exit_with_code
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import PyIRFunctionDef, PyIRNode
from ...ir_compiler import PyIRSymbolRef
from .ctx import AnalysisContext, AnalysisPass


class AnalyzeInlineRecursionPass(AnalysisPass):
    """
    Анализирует граф вызовов функций с inline/with-декораторами
    и запрещает прямую и косвенную рекурсию.
    """

    TARGET_DECORATORS: tuple[str, ...] = (
        "CompilerDirective.inline",
        "CompilerDirective.contextmanager",
    )

    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if ir.path is None:
            return

        st = ctx.inline_recursion_state
        if st.done:
            return

        for fn in cls._iter_module_functions(ir):
            if not cls._has_target_decorator(fn):
                continue

            sid = cls._get_function_symbol_id(fn.name, ir, ctx)
            if sid is None:
                continue

            st.inline_funcs.add(sid)
            st.locations[sid] = (
                ir.path,
                fn.line,
                fn.offset,
            )

        for fn in cls._iter_module_functions(ir):
            fn_sid = cls._get_function_symbol_id(fn.name, ir, ctx)
            if fn_sid is None:
                continue

            if fn_sid not in st.inline_funcs:
                continue

            for callee_sid in cls._collect_called_global_function_symbol_ids(fn.body):
                st.raw_edges.setdefault(fn_sid, set()).add(callee_sid)

        st.processed_files.add(ir.path)
        if len(st.processed_files) >= len(ctx.file_simbol_data):
            cls._finalize_and_check(ctx)

    @classmethod
    def _finalize_and_check(cls, ctx: AnalysisContext) -> None:
        st = ctx.inline_recursion_state
        if st.done:
            return

        inline_funcs: set[int] = st.inline_funcs
        raw_edges: dict[int, set[int]] = st.raw_edges

        graph: dict[int, set[int]] = {}

        for src, dsts in raw_edges.items():
            if src not in inline_funcs:
                continue
            graph[src] = {d for d in dsts if d in inline_funcs}

        for sid in inline_funcs:
            graph.setdefault(sid, set())

        cycle = cls._find_cycle(graph)
        if cycle is not None:
            cls._error(ctx, cycle[0])

        st.done = True

    @staticmethod
    def _find_cycle(graph: dict[int, set[int]]) -> list[int] | None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[int, int] = {n: WHITE for n in graph}
        parent: dict[int, int] = {}

        def dfs(u: int) -> list[int] | None:
            color[u] = GRAY
            for v in graph[u]:
                if color[v] == WHITE:
                    parent[v] = u
                    res = dfs(v)
                    if res is not None:
                        return res

                elif color[v] == GRAY:
                    return [v, u]

            color[u] = BLACK
            return None

        for n in graph:
            if color[n] == WHITE:
                res = dfs(n)
                if res is not None:
                    return res

        return None

    @staticmethod
    def _iter_module_functions(ir: PyIRFile) -> Iterable[PyIRFunctionDef]:
        for node in ir.body:
            if isinstance(node, PyIRFunctionDef):
                yield node

    @classmethod
    def _has_target_decorator(cls, fn: PyIRFunctionDef) -> bool:
        return any(d.name in cls.TARGET_DECORATORS for d in fn.decorators or ())

    @staticmethod
    def _get_function_symbol_id(
        name: str,
        ir: PyIRFile,
        ctx: AnalysisContext,
    ) -> int | None:
        if ir.path is None:
            return None

        for sid, info in ctx.symbols.items():
            if info.file != ir.path:
                continue

            sym = info.symbol
            if sym.scope == "module" and sym.kind == "function" and sym.name == name:
                return sid

        return None

    @classmethod
    def _collect_called_global_function_symbol_ids(
        cls,
        body: list[PyIRNode],
    ) -> set[int]:
        out: set[int] = set()

        for node in cls._walk_nodes(body):
            if node.__class__.__name__ != "PyIRCall":
                continue

            func = getattr(node, "func", None)
            if not isinstance(func, PyIRSymbolRef):
                continue

            if func.sym_id is not None:
                out.add(func.sym_id)

        return out

    @classmethod
    def _walk_nodes(cls, root: Any) -> Iterable[Any]:
        stack: list[Any] = [root]

        while stack:
            cur = stack.pop()

            if isinstance(cur, (list, tuple)):
                stack.extend(cur)
                continue

            if isinstance(cur, dict):
                stack.extend(cur.values())
                continue

            if isinstance(cur, PyIRNode):
                yield cur

            if is_dataclass(cur):
                for f in fields(cur):
                    try:
                        stack.append(getattr(cur, f.name))
                    except Exception:
                        pass

    @staticmethod
    def _error(ctx: AnalysisContext, sid: int) -> None:
        info = ctx.symbols.get(sid)
        loc = ctx.inline_recursion_state.locations.get(sid)

        file = info.file if info else "<unknown>"
        line = loc[1] if loc else None
        offset = loc[2] if loc else None

        exit_with_code(
            1,
            (
                "Обнаружена рекурсия между inline/with-функциями.\n"
                "Рекурсивные inline/with-вызовы запрещены.\n"
                f"Функция: {info.fqname if info else sid}\n"
                f"Файл: {file}\n"
                f"LINE|OFFSET: {line}|{offset}"
            ),
        )
        assert False, "unreachable"

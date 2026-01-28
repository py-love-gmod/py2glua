from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Set

from ....._cli.logging_setup import exit_with_code
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
)
from .symlinks import PyIRSymLink, SymbolId, SymLinkContext

_Directive = Literal["inline", "contextmanager"]


@dataclass(frozen=True)
class _MacroFnInfo:
    sym_id: int
    name: str
    file: Path | None
    line: int | None
    offset: int | None
    kinds: Set[_Directive]


class NoInlineRecursionPass:
    """
    Запрещает прямую и косвенную рекурсию среди функций,
    помеченных @inline() и @contextmanager().
    """

    _CTX_SEEN = "_macrorec_seen_paths"
    _CTX_DONE = "_macrorec_done"
    _CTX_MACRO_FUNCS = "_macrorec_macro_funcs"
    _CTX_EDGES = "_macrorec_edges"

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        NoInlineRecursionPass._ensure_ctx(ctx)

        if ir.path is not None:
            getattr(ctx, NoInlineRecursionPass._CTX_SEEN).add(ir.path.resolve())

        NoInlineRecursionPass._collect_macro_functions(ir, ctx)
        NoInlineRecursionPass._collect_macro_edges(ir, ctx)
        NoInlineRecursionPass._maybe_validate(ctx)

    @staticmethod
    def _ensure_ctx(ctx: SymLinkContext) -> None:
        if not hasattr(ctx, NoInlineRecursionPass._CTX_SEEN):
            setattr(ctx, NoInlineRecursionPass._CTX_SEEN, set())

        if not hasattr(ctx, NoInlineRecursionPass._CTX_DONE):
            setattr(ctx, NoInlineRecursionPass._CTX_DONE, False)

        if not hasattr(ctx, NoInlineRecursionPass._CTX_MACRO_FUNCS):
            setattr(ctx, NoInlineRecursionPass._CTX_MACRO_FUNCS, {})

        if not hasattr(ctx, NoInlineRecursionPass._CTX_EDGES):
            setattr(ctx, NoInlineRecursionPass._CTX_EDGES, {})

    @staticmethod
    def _decorator_kind(expr: PyIRNode) -> _Directive | None:
        if isinstance(expr, PyIRAttribute) and expr.attr in (
            "inline",
            "contextmanager",
        ):
            return expr.attr  # type: ignore[return-value]

        if isinstance(expr, PyIRCall):
            f = expr.func
            if isinstance(f, PyIRAttribute) and f.attr in ("inline", "contextmanager"):
                return f.attr  # type: ignore[return-value]

            if isinstance(f, PyIRSymLink) and f.name in ("inline", "contextmanager"):
                return f.name  # type: ignore[return-value]

        return None

    @staticmethod
    def _func_symbol_id(fn: PyIRFunctionDef, ctx: SymLinkContext) -> int | None:
        decl_scope = ctx.node_scope[ctx.uid(fn)]
        link = ctx.resolve(scope=decl_scope, name=fn.name)
        return None if link is None else link.target.value

    @staticmethod
    def _collect_macro_functions(ir: PyIRFile, ctx: SymLinkContext) -> None:
        macro_funcs: Dict[int, _MacroFnInfo] = getattr(
            ctx, NoInlineRecursionPass._CTX_MACRO_FUNCS
        )

        for fn in ir.walk():
            if not isinstance(fn, PyIRFunctionDef):
                continue

            kinds: Set[_Directive] = set()
            for dec in fn.decorators:
                expr = getattr(dec, "exper", None)
                if expr is None:
                    continue

                k = NoInlineRecursionPass._decorator_kind(expr)
                if k is not None:
                    kinds.add(k)

            if not kinds:
                continue

            sym_id = NoInlineRecursionPass._func_symbol_id(fn, ctx)
            if sym_id is None:
                continue

            sym = ctx.symbols.get(SymbolId(sym_id))

            macro_funcs[sym_id] = _MacroFnInfo(
                sym_id=sym_id,
                name=fn.name,
                file=sym.decl_file if sym else None,
                line=sym.decl_line if sym else fn.line,
                offset=fn.offset,
                kinds=kinds,
            )

    @staticmethod
    def _collect_macro_edges(ir: PyIRFile, ctx: SymLinkContext) -> None:
        macro_funcs: Dict[int, _MacroFnInfo] = getattr(
            ctx, NoInlineRecursionPass._CTX_MACRO_FUNCS
        )
        edges: Dict[int, Set[int]] = getattr(ctx, NoInlineRecursionPass._CTX_EDGES)

        macro_set = set(macro_funcs.keys())
        if not macro_set:
            return

        def walk_calls(node: PyIRNode) -> Iterable[PyIRCall]:
            stack = [node]
            while stack:
                cur = stack.pop()
                if isinstance(cur, PyIRCall):
                    yield cur

                if isinstance(cur, (PyIRFunctionDef, PyIRClassDef)) and cur is not node:
                    continue

                for ch in cur.walk():
                    if ch is not cur:
                        stack.append(ch)

        for fn in ir.walk():
            if not isinstance(fn, PyIRFunctionDef):
                continue

            caller = NoInlineRecursionPass._func_symbol_id(fn, ctx)
            if caller not in macro_set:
                continue

            edges.setdefault(caller, set())

            for call in walk_calls(fn):
                if isinstance(call.func, PyIRSymLink):
                    edges[caller].add(call.func.symbol_id)

    @staticmethod
    def _maybe_validate(ctx: SymLinkContext) -> None:
        if getattr(ctx, NoInlineRecursionPass._CTX_DONE):
            return

        all_paths = {p.resolve() for p in getattr(ctx, "_all_paths", set()) if p}
        seen = getattr(ctx, NoInlineRecursionPass._CTX_SEEN)
        if all_paths and seen != all_paths:
            return

        setattr(ctx, NoInlineRecursionPass._CTX_DONE, True)

        macro_funcs: Dict[int, _MacroFnInfo] = getattr(
            ctx, NoInlineRecursionPass._CTX_MACRO_FUNCS
        )
        edges: Dict[int, Set[int]] = getattr(ctx, NoInlineRecursionPass._CTX_EDGES)

        graph = {
            k: [d for d in edges.get(k, []) if d in macro_funcs] for k in macro_funcs
        }

        cycle = NoInlineRecursionPass._find_cycle(graph)
        if cycle is None:
            return

        first = macro_funcs[cycle[0]]
        chain = " -> ".join(macro_funcs[s].name for s in cycle)

        exit_with_code(
            1,
            f"Обнаружена рекурсия среди {next(iter(first.kinds))} функций\n"
            f"Цепочка: {chain}\n"
            f"Файл: {first.file}\n"
            f"LINE|OFFSET: {first.line}|{first.offset}",
        )

    @staticmethod
    def _find_cycle(graph: Dict[int, List[int]]) -> List[int] | None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in graph}
        parent: Dict[int, int] = {}

        def dfs(v: int) -> List[int] | None:
            color[v] = GRAY
            for to in graph[v]:
                if color[to] == WHITE:
                    parent[to] = v
                    res = dfs(to)
                    if res:
                        return res

                elif color[to] == GRAY:
                    path = [to]
                    cur = v
                    while cur != to:
                        path.append(cur)
                        cur = parent[cur]

                    path.append(to)
                    path.reverse()
                    return path

            color[v] = BLACK
            return None

        for n in graph:
            if color[n] == WHITE:
                res = dfs(n)
                if res:
                    return res

        return None

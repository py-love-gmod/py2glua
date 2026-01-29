from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRCall,
    PyIRFile,
    PyIRFunctionDef,
)
from ..macro_utils import MacroUtils
from .symlinks import PyIRSymLink, SymbolId, SymLinkContext

_DIRECTIVE_KINDS: tuple[str, ...] = (
    "inline",
    "contextmanager",
    "anonymous",
)


@dataclass(frozen=True)
class _MacroFnInfo:
    sym_id: int
    name: str
    file: Path | None
    line: int | None
    offset: int | None
    kinds: set[str]


class NoInlineRecursionPass:
    """
    Запрещает прямую и косвенную рекурсию среди функций,
    помеченных директивами CompilerDirective:
      - inline
      - contextmanager
      - anonymous
    """

    _CTX_SEEN = "_macrorec_seen_paths"
    _CTX_DONE = "_macrorec_done"
    _CTX_MACRO_FUNCS = "_macrorec_macro_funcs"  # sym_id -> _MacroFnInfo
    _CTX_EDGES = "_macrorec_edges"  # sym_id -> set[sym_id]

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        MacroUtils.ensure_pass_ctx(
            ctx,
            **{
                NoInlineRecursionPass._CTX_SEEN: set(),
                NoInlineRecursionPass._CTX_DONE: False,
                NoInlineRecursionPass._CTX_MACRO_FUNCS: {},
                NoInlineRecursionPass._CTX_EDGES: {},
            },
        )

        if ir.path is not None:
            getattr(ctx, NoInlineRecursionPass._CTX_SEEN).add(ir.path.resolve())

        NoInlineRecursionPass._collect_macro_functions(ir, ctx)
        NoInlineRecursionPass._collect_macro_edges(ir, ctx)

        if MacroUtils.ready_to_finalize(
            ctx,
            seen_key=NoInlineRecursionPass._CTX_SEEN,
            done_key=NoInlineRecursionPass._CTX_DONE,
        ):
            NoInlineRecursionPass._validate(ctx)

    @staticmethod
    def _function_symbol_id(fn: PyIRFunctionDef, ctx: SymLinkContext) -> int | None:
        uid = ctx.uid(fn)
        scope = ctx.node_scope[uid]
        link = ctx.resolve(scope=scope, name=fn.name)
        return None if link is None else link.target.value

    @staticmethod
    def _collect_macro_functions(ir: PyIRFile, ctx: SymLinkContext) -> None:
        macro_funcs: Dict[int, _MacroFnInfo] = getattr(
            ctx, NoInlineRecursionPass._CTX_MACRO_FUNCS
        )

        MacroUtils.collect_marked_defs_multi(
            ir,
            ctx,
            node_type=PyIRFunctionDef,
            kinds=_DIRECTIVE_KINDS,
            out=macro_funcs,
            get_symbol_id=NoInlineRecursionPass._function_symbol_id,
            make_info=lambda fn, kinds, sym_id: _MacroFnInfo(
                sym_id=sym_id,
                name=(ctx.symbols.get(SymbolId(sym_id)).name),  # type: ignore
                file=(ctx.symbols.get(SymbolId(sym_id)).decl_file),  # type: ignore
                line=(ctx.symbols.get(SymbolId(sym_id)).decl_line),  # type: ignore
                offset=fn.offset,
                kinds=set(kinds),
            ),
        )

    @staticmethod
    def _collect_macro_edges(ir: PyIRFile, ctx: SymLinkContext) -> None:
        macro_funcs: Dict[int, _MacroFnInfo] = getattr(
            ctx, NoInlineRecursionPass._CTX_MACRO_FUNCS
        )
        edges: Dict[int, set[int]] = getattr(ctx, NoInlineRecursionPass._CTX_EDGES)

        macro_set = set(macro_funcs)
        if not macro_set:
            return

        for fn in ir.walk():
            if not isinstance(fn, PyIRFunctionDef):
                continue

            caller = NoInlineRecursionPass._function_symbol_id(fn, ctx)
            if caller not in macro_set:
                continue

            edges.setdefault(caller, set())

            for node in fn.walk():
                if not isinstance(node, PyIRCall):
                    continue

                if isinstance(node.func, PyIRSymLink):
                    callee = node.func.symbol_id
                    if callee in macro_set:
                        edges[caller].add(callee)

    @staticmethod
    def _validate(ctx: SymLinkContext) -> None:
        macro_funcs: Dict[int, _MacroFnInfo] = getattr(
            ctx, NoInlineRecursionPass._CTX_MACRO_FUNCS
        )
        edges: Dict[int, set[int]] = getattr(ctx, NoInlineRecursionPass._CTX_EDGES)

        graph = {
            k: [d for d in edges.get(k, ()) if d in macro_funcs] for k in macro_funcs
        }

        cycle = NoInlineRecursionPass._find_cycle(graph)
        if cycle is None:
            return

        first = macro_funcs[cycle[0]]
        chain = " -> ".join(macro_funcs[s].name for s in cycle)

        CompilerExit.user_error(
            f"Обнаружена рекурсия среди {next(iter(first.kinds))} функций\n"
            f"Цепочка: {chain}",
            path=first.file,
            line=first.line,
            offset=first.offset,
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

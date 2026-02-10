from __future__ import annotations

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRDecorator,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
    PyIRVarUse,
)
from ..common import CORE_COMPILER_DIRECTIVE_ATTR_PREFIXES

_CT_METHODS = {"inline", "anonymous", "contextmanager"}


class RecursionSanityCheckPass:
    """
    Запрещает прямую и косвенную рекурсию для функций.
    """

    @staticmethod
    def run(ir: PyIRFile) -> None:
        ct_funcs: dict[str, PyIRFunctionDef] = {}

        for node in ir.walk():
            if not isinstance(node, PyIRFunctionDef):
                continue

            if RecursionSanityCheckPass._is_compile_time_fn(node):
                ct_funcs[node.name] = node

        if not ct_funcs:
            return

        graph: dict[str, set[str]] = {name: set() for name in ct_funcs}

        for name, fn in ct_funcs.items():
            called: set[str] = set()
            for st in fn.body:
                RecursionSanityCheckPass._collect_calls_to_ct_funcs(
                    st, out=called, ct_names=set(ct_funcs.keys())
                )

            graph[name] = called

        RecursionSanityCheckPass._fail_on_cycle(ir, graph, ct_funcs)

    @staticmethod
    def _is_compile_time_fn(fn: PyIRFunctionDef) -> bool:
        for dec in fn.decorators:
            kind = RecursionSanityCheckPass._compiler_directive_kind(dec)
            if kind in _CT_METHODS:
                return True

        return False

    @staticmethod
    def _compiler_directive_kind(dec: PyIRDecorator) -> str | None:
        expr = dec.exper

        if isinstance(expr, PyIRAttribute):
            if RecursionSanityCheckPass._is_compiler_directive_attr(expr):
                return expr.attr

            return None

        if isinstance(expr, PyIRCall):
            f = expr.func
            if isinstance(f, PyIRAttribute):
                if RecursionSanityCheckPass._is_compiler_directive_attr(f):
                    return f.attr

            return None

        return None

    @staticmethod
    def _is_compiler_directive_attr(attr: PyIRAttribute) -> bool:
        if attr.attr not in _CT_METHODS:
            return False

        parts = RecursionSanityCheckPass._attr_chain_parts(attr.value)
        if parts is None:
            return False

        return tuple(parts) in CORE_COMPILER_DIRECTIVE_ATTR_PREFIXES

    @staticmethod
    def _attr_chain_parts(node: PyIRNode) -> list[str] | None:
        names_rev: list[str] = []
        cur: PyIRNode = node

        while isinstance(cur, PyIRAttribute):
            names_rev.append(cur.attr)
            cur = cur.value

        if not isinstance(cur, PyIRVarUse):
            return None

        names_rev.append(cur.name)
        names_rev.reverse()
        return names_rev

    @staticmethod
    def _collect_calls_to_ct_funcs(
        node: PyIRNode, *, out: set[str], ct_names: set[str]
    ) -> None:
        if isinstance(node, PyIRCall):
            f = node.func
            if isinstance(f, PyIRVarUse) and f.name in ct_names:
                out.add(f.name)

        for ch in node.walk():
            if ch is node:
                continue

            RecursionSanityCheckPass._collect_calls_to_ct_funcs(
                ch, out=out, ct_names=ct_names
            )

    @staticmethod
    def _fail_on_cycle(
        ir: PyIRFile,
        graph: dict[str, set[str]],
        defs: dict[str, PyIRFunctionDef],
    ) -> None:
        UNVISITED, VISITING, DONE = 0, 1, 2
        state: dict[str, int] = {k: UNVISITED for k in graph}

        stack: list[str] = []
        index_in_stack: dict[str, int] = {}

        def fmt_decl(name: str) -> str:
            fn = defs.get(name)
            if fn is None:
                return name

            line = fn.line
            off = fn.offset
            if line is None and off is None:
                return name

            return f"{name} (line={line}, offset={off})"

        def fail_cycle(start_name: str) -> None:
            i = index_in_stack.get(start_name, 0)
            ring = stack[i:] + [start_name]

            chain = "\n".join(
                f"  {fmt_decl(ring[j])} -> {fmt_decl(ring[j + 1])}"
                for j in range(len(ring) - 1)
            )

            CompilerExit.user_error(
                "Для директив inline / anonymous / contextmanager рекурсия запрещена.\n"
                "Цепочка вызовов:\n"
                f"{chain}",
                path=ir.path,
                show_pos=False,
            )

        def dfs(v: str) -> None:
            state[v] = VISITING
            index_in_stack[v] = len(stack)
            stack.append(v)

            for u in graph[v]:
                if state[u] == VISITING:
                    fail_cycle(u)

                if state[u] == UNVISITED:
                    dfs(u)

            stack.pop()
            index_in_stack.pop(v, None)
            state[v] = DONE

        for v in graph:
            if state[v] == UNVISITED:
                dfs(v)

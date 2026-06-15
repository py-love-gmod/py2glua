from __future__ import annotations

from pathlib import Path

from plg_reader import IRFile, IRImport

from ..utils import Shutdown


class DependencyGraphConstructor:
    @staticmethod
    def _get_module_name(file_path: Path) -> str:
        parts = list(file_path.parts)
        suffix = parts[-1].split(".")[-1]
        if suffix in ("py", "pyi"):
            parts[-1] = parts[-1][: -len(suffix) - 1]

        if parts and parts[-1] == "__init__":
            parts.pop()

        return ".".join(parts)

    @classmethod
    def _build_dependency_graph(
        cls,
        irs: dict[str, IRFile],
    ) -> tuple[
        dict[str, list[str]],
        list[list[str]],
    ]:
        module_to_path: dict[str, str] = {}
        real_path_of: dict[str, str] = {}

        for path_str, ir_file in irs.items():
            real_path = Path(ir_file.path)
            mod_name = cls._get_module_name(real_path)
            module_to_path[mod_name] = path_str
            real_path_of[path_str] = str(real_path)

        graph: dict[str, list[str]] = {}
        edge_imports: dict[tuple[str, str], IRImport] = {}

        for path_str, ir_file in irs.items():
            deps = []
            for imp in ir_file.imports:
                if imp.modules:
                    dep_module = imp.modules[0]
                    dep_path = module_to_path.get(dep_module)
                    if dep_path is not None:
                        deps.append(dep_path)
                        edge_imports[(path_str, dep_path)] = imp

            graph[path_str] = deps

        WHITE, GRAY, BLACK = 0, 1, 2
        colors: dict[str, int] = {node: WHITE for node in graph}
        cycles: list[list[str]] = []
        stack: list[str] = []

        def dfs(node: str):
            colors[node] = GRAY
            stack.append(node)
            for neighbor in graph.get(node, []):
                if colors[neighbor] == GRAY:
                    start = stack.index(neighbor)
                    cycle_paths = stack[start:] + [neighbor]
                    steps = []
                    for i in range(len(cycle_paths) - 1):
                        from_key = cycle_paths[i]
                        to_key = cycle_paths[i + 1]
                        imp = edge_imports.get((from_key, to_key))
                        if imp:
                            line = imp.pos[0]
                            module_name = imp.modules[0] if imp.modules else "?"
                            from_real = real_path_of.get(from_key, from_key)
                            steps.append(f"{from_real}:{line}: import {module_name}")

                        else:
                            steps.append(f"{from_key} -> {to_key} (неизвестный импорт)")

                    cycles.append(steps)
                elif colors[neighbor] == WHITE:
                    dfs(neighbor)

            stack.pop()
            colors[node] = BLACK

        for node in graph:
            if colors[node] == WHITE:
                dfs(node)

        return graph, cycles

    @staticmethod
    def _format_cycles(cycles: list[list[str]]) -> str | None:
        if not cycles:
            return None

        lines = ["Обнаружены циклические зависимости:"]
        for i, cycle in enumerate(cycles, 1):
            if i > 1:
                lines.append("")

            lines.append(f"Цикл #{i}:")
            for step in cycle:
                lines.append(f"\t{step}")

        return "\n".join(lines)

    @classmethod
    def build(cls, irs: dict[str, IRFile]) -> dict[str, list[str]]:
        graph, cycles = cls._build_dependency_graph(irs)

        if cycles_str := cls._format_cycles(cycles):
            Shutdown.user_error(cycles_str)

        return graph

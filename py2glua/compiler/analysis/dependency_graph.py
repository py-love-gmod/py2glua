from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plg_reader import IRFile

from utils import Shutdown


@dataclass
class DependencyGraph:
    """
    Ориентированный граф зависимостей между файлами.
    """

    nodes: set[str] = field(default_factory=set)
    """все файлы проекта"""
    forward: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    """a -> b (a импортирует b)"""
    reverse: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    """b -> a (кто импортирует b)"""
    import_positions: dict[tuple[str, str], tuple[int, int]] = field(
        default_factory=dict
    )
    """(from, to) -> (строка, колонка)"""

    def add_node(self, node: str) -> None:
        """Добавляет вершину в граф (если её ещё нет)."""
        self.nodes.add(node)

    def add_edge(self, from_file: str, to_file: str, pos: tuple[int, int]) -> None:
        """Добавляет ориентированное ребро и обе вершины."""
        self.add_node(from_file)
        self.add_node(to_file)
        self.forward[from_file].add(to_file)
        self.reverse[to_file].add(from_file)
        self.import_positions[(from_file, to_file)] = pos

    @classmethod
    def build(cls, irs: dict[str, IRFile], root_path: Path) -> DependencyGraph:
        """
        Строит граф на основе распарсенных файлов.

        Args:
            irs: словарь относительный_путь -> IRFile
            root_path: корневая директория проекта
        """
        graph = cls()

        for rel_path in irs:
            graph.add_node(rel_path)

        for rel_path, ir_file in irs.items():
            current_abs = ir_file.path
            for imp in ir_file.imports:
                if imp.is_from and "*" in imp.names:
                    continue

                for module_name in imp.modules:
                    target_path = cls._resolve_import(
                        module_name,
                        current_abs,
                        root_path,
                        imp.level,
                    )
                    if target_path is None:
                        continue

                    try:
                        target_rel = target_path.relative_to(root_path).as_posix()

                    except ValueError:
                        continue

                    graph.add_edge(rel_path, target_rel, imp.pos)

        return graph

    @staticmethod
    def _resolve_import(
        module_name: str,
        current_file: Path,
        root: Path,
        level: int,
    ) -> Path | None:
        """
        Преобразует импорт в путь к файлу (абсолютный).
        Возвращает None, если файл не найден внутри root.
        """
        if level > 0:
            base = current_file.parent
            for _ in range(level - 1):
                base = base.parent

            if module_name:
                parts = module_name.split(".")
                target = base.joinpath(*parts).with_suffix(".py")

            else:
                target = base / "__init__.py"
                if not target.exists():
                    target = base

            if target.exists() and (target.is_file() or target.is_dir()):
                return target

            return None

        else:
            parts = module_name.split(".")
            target = root
            for part in parts:
                target = target / part

            if target.with_suffix(".py").exists():
                return target.with_suffix(".py")

            if (target / "__init__.py").exists():
                return target

            return None

    def check_cycles(self) -> None:
        """Проверяет наличие циклов; при обнаружении вызывает Shutdown.user_error."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in self.nodes}
        parent = {}

        def dfs(v: str) -> None:
            color[v] = GRAY
            for u in self.forward.get(v, []):
                if color[u] == WHITE:
                    parent[u] = v
                    dfs(u)

                elif color[u] == GRAY:
                    path = []
                    cur = v
                    while cur != u:
                        path.append(cur)
                        cur = parent.get(cur)
                        if cur is None:
                            break

                    else:
                        path.append(u)
                        path.reverse()
                        cycle = path + [path[0]]
                        lines = ["Обнаружен циклический импорт:"]
                        for i in range(len(cycle) - 1):
                            a, b = cycle[i], cycle[i + 1]
                            pos = self.import_positions.get((a, b))
                            if pos:
                                lines.append(f"  {a} -> {b}  (строка {pos[0]})")
                            else:
                                lines.append(f"  {a} -> {b}")

                        Shutdown.user_error("\n".join(lines))

            color[v] = BLACK

        for node in self.forward:
            if color[node] == WHITE:
                dfs(node)

    def get_affected_files(self, changed_file: str) -> set[str]:
        """
        Возвращает множество файлов, которые прямо или косвенно зависят от changed_file.
        """
        affected = set()
        stack = [changed_file]
        while stack:
            node = stack.pop()
            for dependent in self.reverse.get(node, []):
                if dependent not in affected:
                    affected.add(dependent)
                    stack.append(dependent)

        return affected

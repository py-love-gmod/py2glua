from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque

from plg_reader import IRFile, build_python_file

from .shared import module_name_from_relative_path, resolve_target_module


def _find_module_file(
    search_roots: list[Path], module_name: str
) -> tuple[Path, Path] | None:
    rel = module_name.replace(".", "/")
    candidates = [
        rel + ".py",
        rel + ".pyi",
        rel + "/__init__.py",
        rel + "/__init__.pyi",
    ]
    for root in search_roots:
        for cand in candidates:
            full = root / cand
            if full.is_file():
                return root, full

    return None


class ImportCollector:
    @staticmethod
    def collect(
        initial_irs: dict[str, IRFile],
        search_roots: list[Path],
        builtin_modules: frozenset[str] = frozenset(),
    ) -> dict[str, IRFile]:
        """Собирает недостающие модули рекурсивно из search_roots."""
        if not initial_irs:
            return {}

        resolved = dict(initial_irs)
        queue: Deque[str] = deque(resolved.keys())
        processed: set[str] = set()

        while queue:
            rel_key = queue.popleft()
            if rel_key in processed:
                continue

            processed.add(rel_key)
            ir_file = resolved[rel_key]
            current_module = module_name_from_relative_path(rel_key)

            for imp in ir_file.imports:
                target_module = resolve_target_module(imp, current_module)

                if target_module in builtin_modules:
                    continue

                if any(
                    module_name_from_relative_path(k) == target_module for k in resolved
                ):
                    continue

                found = _find_module_file(search_roots, target_module)
                if found is None:
                    continue

                root, abs_path = found
                ir = build_python_file(abs_path, strip_comments=False)
                new_rel = abs_path.relative_to(root).as_posix()
                resolved[new_rel] = ir
                queue.append(new_rel)

        return resolved

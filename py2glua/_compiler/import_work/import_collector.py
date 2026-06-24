from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque

from plg_reader import IRFile, build_python_file

from ..shared import module_name_from_relative_path, resolve_target_module


def _find_module(root: Path, rel_path: str) -> Path | None:
    if rel_path:
        candidates = [
            rel_path + ".py",
            rel_path + ".pyi",
            rel_path + "/__init__.py",
            rel_path + "/__init__.pyi",
        ]
    else:
        candidates = ["__init__.py", "__init__.pyi"]

    for cand in candidates:
        full = root / cand
        if full.is_file():
            return full

    return None


def _resolve_module_path(
    target_module: str,
    search_roots: list[Path],
    prefix_roots: dict[str, Path],
) -> tuple[Path, str] | None:
    for prefix, root in prefix_roots.items():
        if not target_module.startswith(prefix):
            continue
        suffix = target_module[len(prefix) :]
        if suffix.startswith("."):
            suffix = suffix[1:]

        rel = suffix.replace(".", "/")
        abs_path = _find_module(root, rel)
        if abs_path is None:
            continue

        prefix_path = prefix.replace(".", "/")
        if rel:
            if abs_path.name in ("__init__.py", "__init__.pyi"):
                key = f"{prefix_path}/{rel}/__init__.py"

            else:
                key = f"{prefix_path}/{rel}.py"

        else:
            key = f"{prefix_path}/__init__.py"

        return abs_path, key

    rel = target_module.replace(".", "/")
    for root in search_roots:
        abs_path = _find_module(root, rel)
        if abs_path is not None:
            key = abs_path.relative_to(root).as_posix()
            return abs_path, key

    return None


class ImportCollector:
    @staticmethod
    def collect(
        initial_irs: dict[str, IRFile],
        search_roots: list[Path],
        builtin_modules: frozenset[str] = frozenset(),
        prefix_roots: dict[str, Path] | None = None,
    ) -> dict[str, IRFile]:
        if not initial_irs:
            return {}

        prefix_roots = prefix_roots or {}
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
            is_pkg = rel_key.endswith("__init__.py")

            for imp in ir_file.imports:
                target_module = resolve_target_module(
                    imp, current_module, is_package=is_pkg
                )

                if target_module in builtin_modules:
                    continue

                if any(
                    module_name_from_relative_path(k) == target_module for k in resolved
                ):
                    continue

                result = _resolve_module_path(target_module, search_roots, prefix_roots)
                if result is None:
                    continue

                abs_path, new_rel = result
                ir = build_python_file(abs_path, strip_comments=False)
                resolved[new_rel] = ir
                queue.append(new_rel)

        return resolved

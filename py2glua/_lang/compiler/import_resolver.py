from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..._cli import CompilerExit
from ..py.ir_builder import PyIRBuilder, PyIRFile
from ..py.ir_dataclass import (
    PyIRAssign,
    PyIRClassDef,
    PyIRFunctionDef,
    PyIRImport,
    PyIRImportType,
    PyIRVarCreate,
    PyIRVarUse,
)


@dataclass(frozen=True)
class _Resolved:
    itype: PyIRImportType
    deps: tuple[Path, ...]


class _SymbolIndex:
    def __init__(self, root: Path):
        self._root = root.resolve()
        self._built = False
        self._sym2path: dict[str, Path] = {}

    def ensure_built(self) -> None:
        if self._built:
            return

        if not self._root.exists():
            self._built = True
            return

        for path in self._root.rglob("*.py"):
            if not path.is_file():
                continue

            try:
                src = path.read_text(encoding="utf-8")
                ir = PyIRBuilder.build_file(source=src, path_to_file=path)
            except Exception:
                # Индекс best-effort: не валим сборку из-за одного файла
                continue

            self._index_file(ir)

        self._built = True

    def resolve(self, name: str) -> Path | None:
        self.ensure_built()
        return self._sym2path.get(name)

    def _index_file(self, ir: PyIRFile) -> None:
        for node in ir.body or []:
            if isinstance(node, (PyIRClassDef, PyIRFunctionDef, PyIRVarCreate)):
                self._add(node.name, ir.path)

            elif isinstance(node, PyIRAssign):
                for t in node.targets:
                    if isinstance(t, PyIRVarUse):
                        self._add(t.name, ir.path)

    def _add(self, name: str, path: Path | None) -> None:
        if not path or not name or name.startswith("_"):
            return
        self._sym2path.setdefault(name, path.resolve())


class ImportResolver:
    _INTERNAL_PREFIX = ("py2glua", "glua")

    def __init__(
        self,
        *,
        project_root: Path,
        internal_root: Path,
    ):
        self.project_root = project_root.resolve()
        self.internal_root = internal_root.resolve()

        self._project_index = _SymbolIndex(self.project_root)
        self._internal_index = _SymbolIndex(self.internal_root)

    def collect_deps(self, *, ir: PyIRFile, current_file: Path) -> list[Path]:
        deps: set[Path] = set()

        for imp in self._collect_imports(ir):
            resolved = self.classify_and_resolve(imp=imp, current_file=current_file)
            imp.itype = resolved.itype

            if resolved.itype == PyIRImportType.EXTERNAL:
                CompilerExit.user_error(
                    "Внешний импорт не поддерживается\n"
                    f"Импорт: {'.'.join(imp.modules or [])}",
                    path=current_file,
                    show_pos=False,
                )

            deps.update(resolved.deps)

        return sorted(p.resolve() for p in deps)

    def classify_and_resolve(self, *, imp: PyIRImport, current_file: Path) -> _Resolved:
        self._reject_star_import(imp, current_file)

        modules = imp.modules or []

        top = modules[0] if modules else None
        if top is None:
            top = self._best_effort_top_name(imp)

        if top and self._is_stdlib_module(top):
            return _Resolved(PyIRImportType.STD_LIB, ())

        if imp.level:
            deps = self._resolve_relative(imp, current_file)
            if not deps:
                CompilerExit.user_error(
                    "Не удалось разрешить относительный импорт\n"
                    f"Импорт: {'.'.join(modules)}",
                    path=current_file,
                )

            d0 = deps[0]
            if self._is_within(d0, self.project_root):
                return _Resolved(PyIRImportType.LOCAL, tuple(deps))

            if self._is_within(d0, self.internal_root):
                return _Resolved(PyIRImportType.INTERNAL, tuple(deps))

            return _Resolved(PyIRImportType.EXTERNAL, ())

        deps_local = self._resolve_absolute(self.project_root, imp, modules)
        if deps_local:
            return _Resolved(PyIRImportType.LOCAL, tuple(deps_local))

        if self._is_internal_namespace(modules):
            deps_internal = self._resolve_internal(imp, modules, current_file)
            if deps_internal:
                return _Resolved(PyIRImportType.INTERNAL, tuple(deps_internal))

            return _Resolved(PyIRImportType.INTERNAL, ())

        return _Resolved(PyIRImportType.EXTERNAL, ())

    def _resolve_internal(
        self,
        imp: PyIRImport,
        modules: list[str],
        current_file: Path,
    ) -> list[Path]:
        """
        INTERNAL namespace: py2glua.glua.<rest...>
        """

        rest = modules[2:]

        # import py2glua.glua.ent
        if not imp.if_from:
            p = self._resolve_module_or_package(self.internal_root, rest)
            return [p] if p else []

        # from py2glua.glua.ent import ENT
        parent = self._resolve_module_or_package(self.internal_root, rest)
        if parent is None:
            return []

        # КЛЮЧЕВАЯ ПРАВКА:
        # Если parent — это модуль (НЕ пакет), то для deps достаточно самого parent.
        # Никаких попыток "разрешить символ ENT" делать не надо.
        if parent.name != "__init__.py":
            return [parent]

        # parent — пакет: from pkg import X -> X может быть подмодулем/пакетом или символом.
        pkg_dir = parent.parent
        deps: set[Path] = {parent}

        for name in self._iter_imported_names(imp):
            p = self._resolve_module_or_package(pkg_dir, [name])
            if p:
                deps.add(p)
                continue

            p2 = self._internal_index.resolve(name)
            if p2:
                deps.add(p2)
                continue

            CompilerExit.user_error(
                f"Не удалось разрешить internal-символ {name}",
                path=current_file,
                show_pos=False,
            )

        return sorted(deps)

    def _resolve_absolute(
        self,
        base: Path,
        imp: PyIRImport,
        modules: list[str],
    ) -> list[Path]:
        if not modules:
            return []

        if not imp.if_from:
            p = self._resolve_module_or_package(base, modules)
            return [p] if p else []

        parent = self._resolve_module_or_package(base, modules)
        if parent is None:
            return []

        if parent.name != "__init__.py":
            return [parent]

        pkg_dir = parent.parent
        deps: set[Path] = {parent}

        for name in self._iter_imported_names(imp):
            p = self._resolve_module_or_package(pkg_dir, [name])
            if p:
                deps.add(p)
                continue

            p2 = self._project_index.resolve(name)
            if p2:
                deps.add(p2)
                continue

            CompilerExit.user_error(
                f"Не удалось разрешить импорт {name}",
                path=parent,
                show_pos=False,
            )

        return sorted(deps)

    def _resolve_relative(
        self,
        imp: PyIRImport,
        current_file: Path,
    ) -> list[Path]:
        base = current_file.parent
        for _ in range(imp.level - 1):
            base = base.parent

        modules = imp.modules or []

        if not imp.if_from:
            p = self._resolve_module_or_package(base, modules)
            return [p] if p else []

        parent = self._resolve_module_or_package(base, modules)
        if parent is None:
            return []

        if parent.name != "__init__.py":
            return [parent]

        pkg_dir = parent.parent
        deps: set[Path] = {parent}

        for name in self._iter_imported_names(imp):
            p = self._resolve_module_or_package(pkg_dir, [name])
            if p:
                deps.add(p)
                continue

            p2 = self._internal_index.resolve(name)
            if p2:
                deps.add(p2)
                continue

            p3 = self._project_index.resolve(name)
            if p3:
                deps.add(p3)
                continue

            CompilerExit.user_error(
                f"Не удалось разрешить относительный импорт {name}",
                path=current_file,
                show_pos=False,
            )

        return sorted(deps)

    @staticmethod
    def _collect_imports(ir: PyIRFile) -> list[PyIRImport]:
        return [n for n in ir.walk() if isinstance(n, PyIRImport)]

    @staticmethod
    def _iter_imported_names(imp: PyIRImport) -> Iterable[str]:
        for n in imp.names or []:
            yield n[0] if isinstance(n, tuple) else n

    @staticmethod
    def _resolve_module_or_package(base: Path, parts: list[str]) -> Path | None:
        if not parts:
            p = base / "__init__.py"
            return p if p.exists() else None

        mod = base.joinpath(*parts).with_suffix(".py")
        if mod.exists():
            return mod

        pkg = base.joinpath(*parts, "__init__.py")
        if pkg.exists():
            return pkg

        return None

    @staticmethod
    def _reject_star_import(imp: PyIRImport, current_file: Path) -> None:
        for name in imp.names or []:
            n = name[0] if isinstance(name, tuple) else name
            if n == "*":
                CompilerExit.user_error_node(
                    "import * не поддерживается",
                    path=current_file,
                    node=imp,
                )

    def _is_internal_namespace(self, modules: list[str]) -> bool:
        return len(modules) >= 2 and (modules[0], modules[1]) == self._INTERNAL_PREFIX

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_stdlib_module(name: str) -> bool:
        return name in sys.stdlib_module_names or name in sys.builtin_module_names

    @staticmethod
    def _best_effort_top_name(imp: PyIRImport) -> str | None:
        if imp.modules:
            return imp.modules[0]

        if imp.names:
            first = imp.names[0]
            cand = first[0] if isinstance(first, tuple) else first
            if isinstance(cand, str) and cand:
                return cand

        return None

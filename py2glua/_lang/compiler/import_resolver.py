from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..._cli.logging_setup import exit_with_code
from ..py.ir_builder import PyIRBuilder, PyIRFile
from ..py.ir_dataclass import (
    PyIRAssign,
    PyIRClassDef,
    PyIRFunctionDef,
    PyIRImport,
    PyIRImportType,
    PyIRVarUse,
)


@dataclass(frozen=True)
class _Resolved:
    itype: PyIRImportType
    deps: tuple[Path, ...]


class _InternalSymbolIndex:
    def __init__(self, internal_root: Path):
        self._root = internal_root.resolve()
        self._built = False
        self._sym2path: dict[str, Path] = {}

    def ensure_built(self) -> None:
        if self._built:
            return

        if not self._root.exists():
            self._built = True
            return

        for path in self._root.rglob("*.py"):
            if not path.is_file() or path.name == "__init__.py":
                continue

            try:
                src = path.read_text(encoding="utf-8")
                ir = PyIRBuilder.build_file(source=src, path_to_file=path)

            except Exception:
                continue

            self._index_file(ir)

        self._built = True

    def resolve(self, name: str) -> Path | None:
        self.ensure_built()
        return self._sym2path.get(name)

    def _index_file(self, ir: PyIRFile) -> None:
        for node in ir.body or []:
            if isinstance(node, (PyIRClassDef, PyIRFunctionDef)):
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
    def __init__(
        self,
        *,
        project_root: Path,
        internal_root: Path,
        internal_prefix: tuple[str, str],
    ):
        self.project_root = project_root.resolve()
        self.internal_root = internal_root.resolve()
        self.internal_prefix = internal_prefix
        self._internal_index = _InternalSymbolIndex(self.internal_root)

    def _fmt_file(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))

        except Exception:
            return str(path)

    def collect_deps(self, *, ir: PyIRFile, current_file: Path) -> list[Path]:
        deps: set[Path] = set()

        for imp in self._collect_imports(ir):
            resolved = self.classify_and_resolve(imp=imp, current_file=current_file)
            imp.itype = resolved.itype

            if resolved.itype == PyIRImportType.EXTERNAL:
                exit_with_code(
                    1,
                    "Внешний импорт не поддерживается\n"
                    f"Импорт: {'.'.join(imp.modules or [])}\n"
                    f"Файл: {self._fmt_file(current_file)}",
                )
                raise AssertionError("unreachable")

            deps.update(resolved.deps)

        return sorted(p.resolve() for p in deps)

    def classify_and_resolve(self, *, imp: PyIRImport, current_file: Path) -> _Resolved:
        self._reject_star_import(imp, current_file)

        modules = imp.modules or []
        top = modules[0] if modules else None

        if imp.level:
            deps = self._resolve_relative(imp, current_file)
            if not deps:
                exit_with_code(
                    1,
                    "Не удалось разрешить относительный импорт\n"
                    f"Импорт: {'.'.join(modules)}\n"
                    f"Файл: {self._fmt_file(current_file)}",
                )
                raise AssertionError("unreachable")

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
            return _Resolved(
                PyIRImportType.INTERNAL,
                tuple(self._resolve_internal(imp, modules, current_file)),
            )

        if top and self._is_stdlib_module(top):
            return _Resolved(PyIRImportType.STD_LIB, ())

        return _Resolved(PyIRImportType.EXTERNAL, ())

    def _resolve_internal(
        self,
        imp: PyIRImport,
        modules: list[str],
        current_file: Path,
    ) -> list[Path]:
        rest = modules[2:]

        if not imp.if_from:
            p = self._resolve_module_or_package(self.internal_root, rest)
            return [p] if p else []

        deps: set[Path] = set()
        for name in self._iter_imported_names(imp):
            p = self._resolve_module_or_package(self.internal_root, rest + [name])
            if p:
                deps.add(p)
                continue

            p2 = self._internal_index.resolve(name)
            if p2:
                deps.add(p2)
                continue

            exit_with_code(
                1,
                "Не удалось разрешить internal-символ\n"
                f"Импорт: from {'.'.join(modules)} import {name}\n"
                f"Файл: {self._fmt_file(current_file)}",
            )
            raise AssertionError("unreachable")

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
        deps: set[Path] = set()

        for name in self._iter_imported_names(imp):
            p = self._resolve_module_or_package(pkg_dir, [name])
            if not p:
                exit_with_code(
                    1,
                    "Не удалось разрешить импорт\n"
                    f"Импорт: from {pkg_dir.name} import {name}\n"
                    f"Файл: {self._fmt_file(parent)}",
                )
                raise AssertionError("unreachable")

            deps.add(p)

        return sorted(deps)

    def _resolve_relative(
        self,
        imp: PyIRImport,
        current_file: Path,
    ) -> list[Path]:
        if self._is_within(current_file, self.internal_root):
            base_root = self.internal_root

        else:
            base_root = self.project_root

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
        deps: set[Path] = set()

        for name in self._iter_imported_names(imp):
            p = self._resolve_module_or_package(pkg_dir, [name])
            if not p:
                if self._is_within(base_root, self.internal_root):
                    p2 = self._internal_index.resolve(name)
                    if p2:
                        deps.add(p2)
                        continue

                exit_with_code(
                    1,
                    "Не удалось разрешить относительный импорт\n"
                    f"Импорт: {name}\n"
                    f"Файл: {self._fmt_file(current_file)}",
                )
                raise AssertionError("unreachable")

            deps.add(p)

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
            init_py = base / "__init__.py"
            return init_py.resolve() if init_py.exists() else None

        mod = base.joinpath(*parts).with_suffix(".py")
        if mod.exists():
            return mod.resolve()

        pkg = base.joinpath(*parts, "__init__.py")
        if pkg.exists():
            return pkg.resolve()

        return None

    def _reject_star_import(self, imp: PyIRImport, current_file: Path) -> None:
        for name in self._iter_imported_names(imp):
            if name == "*":
                exit_with_code(
                    1,
                    "import * не поддерживается в py2glua\n"
                    f"Файл: {self._fmt_file(current_file)}",
                )
                raise AssertionError("unreachable")

    def _is_internal_namespace(self, modules: list[str]) -> bool:
        return len(modules) >= 2 and (modules[0], modules[1]) == self.internal_prefix

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

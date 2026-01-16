from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import resources
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
        self._root = internal_root
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


class _ImportResolver:
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

    def collect_deps(self, *, ir: PyIRFile, current_file: Path) -> list[Path]:
        deps: set[Path] = set()

        for imp in self._collect_imports(ir):
            resolved = self.classify_and_resolve(imp=imp, current_file=current_file)
            imp.itype = resolved.itype

            if resolved.itype == PyIRImportType.EXTERNAL:
                exit_with_code(
                    1,
                    "Внешний импорт не поддерживается:\n"
                    f"    {'.'.join(imp.modules or [])}\n"
                    f"Файл: {current_file}",
                )
                raise AssertionError("unreachable")

            deps.update(resolved.deps)

        return sorted(p.resolve() for p in deps)

    def classify_and_resolve(self, *, imp: PyIRImport, current_file: Path) -> _Resolved:
        self._reject_star_import(imp, current_file)

        modules = imp.modules or []
        top = modules[0] if modules else None

        if imp.level:
            deps = self._resolve_relative(imp=imp, current_file=current_file)
            if not deps:
                exit_with_code(
                    1,
                    f"Не удалось разрешить относительный импорт в {current_file}: {modules}",
                )
                raise AssertionError("unreachable")

            any_dep = deps[0]
            if self._is_within(any_dep, self.project_root):
                return _Resolved(PyIRImportType.LOCAL, tuple(deps))

            if self._is_within(any_dep, self.internal_root):
                return _Resolved(PyIRImportType.INTERNAL, tuple(deps))

            return _Resolved(PyIRImportType.EXTERNAL, ())

        deps_local = self._resolve_absolute(self.project_root, imp, modules)
        if deps_local:
            return _Resolved(PyIRImportType.LOCAL, tuple(deps_local))

        if self._is_internal_namespace(modules):
            deps_int = self._resolve_internal(imp, modules)
            return _Resolved(PyIRImportType.INTERNAL, tuple(deps_int))

        if top and self._is_stdlib_module(top):
            return _Resolved(PyIRImportType.STD_LIB, ())

        return _Resolved(PyIRImportType.EXTERNAL, ())

    def _resolve_internal(self, imp: PyIRImport, modules: list[str]) -> list[Path]:
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
                f"Не удалось разрешить internal-символ '{name}'\n"
                f"Импорт: {'.'.join(modules)}\n"
                f"Файл: {imp}",
            )
            raise AssertionError("unreachable")

        return sorted(deps)

    def _resolve_absolute(
        self, base: Path, imp: PyIRImport, modules: list[str]
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
                    f"Не удалось разрешить импорт '{name}' из {pkg_dir}\nФайл: {imp}",
                )
                raise AssertionError("unreachable")

            deps.add(p)

        return sorted(deps)

    def _resolve_relative(self, *, imp: PyIRImport, current_file: Path) -> list[Path]:
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
                exit_with_code(
                    1,
                    f"Не удалось разрешить относительный импорт '{name}'\n"
                    f"Файл: {current_file}",
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
                    f"Файл: {current_file}\n"
                    f"Импорт: from {'.'.join(imp.modules or [])} import *",
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


class Compiler:
    file_passes: list = []
    project_passes: list = []

    _INTERNAL_PREFIX = ("py2glua", "glua")
    _INTERNAL_PKG = "py2glua.glua"

    @classmethod
    def build(cls, project_root: Path) -> list[PyIRFile]:
        root = project_root.resolve()
        if not root.exists():
            exit_with_code(1, f"Папка проекта не найдена: {root}")

        if not root.is_dir():
            exit_with_code(1, f"Указанный путь не является папкой: {root}")

        project_files = sorted(p.resolve() for p in root.rglob("*.py") if p.is_file())
        if not project_files:
            exit_with_code(1, f"В проекте нет ни одного .py файла: {root}")

        try:
            internal_tr = resources.files(cls._INTERNAL_PKG)

        except Exception as e:
            exit_with_code(
                3, f"Не удалось найти internal-пакет {cls._INTERNAL_PKG}: {e}"
            )
            raise AssertionError("unreachable")

        with resources.as_file(internal_tr) as internal_root:
            internal_root = Path(internal_root).resolve()
            resolver = _ImportResolver(
                project_root=root,
                internal_root=internal_root,
                internal_prefix=cls._INTERNAL_PREFIX,
            )

            VISITING = 1
            DONE = 2

            state: dict[Path, int] = {}
            ir_map: dict[Path, PyIRFile] = {}

            index_in_stack: dict[Path, int] = {}
            call_stack: list[Path] = []

            frames: list[tuple[Path, list[Path], int, bool]] = []

            for entry in project_files:
                if entry in state:
                    continue

                frames.append((entry, [], 0, False))

                while frames:
                    path, deps, i, entered = frames.pop()

                    st = state.get(path)
                    if st == DONE:
                        continue

                    if not entered:
                        if st == VISITING:
                            exit_with_code(
                                3,
                                f"Обнаружен внутренний сбой обхода зависимостей: {path}",
                            )
                            raise AssertionError("unreachable")

                        state[path] = VISITING
                        index_in_stack[path] = len(call_stack)
                        call_stack.append(path)

                        ir = cls._build_one(path)
                        ir = cls._run_file_passes(ir)
                        ir_map[path] = ir

                        deps = resolver.collect_deps(ir=ir, current_file=path)

                        frames.append((path, deps, 0, True))
                        continue

                    if i >= len(deps):
                        state[path] = DONE
                        call_stack.pop()
                        index_in_stack.pop(path, None)
                        continue

                    dep = deps[i]
                    frames.append((path, deps, i + 1, True))

                    dep_state = state.get(dep)
                    if dep_state == VISITING:
                        cycle = cls._format_cycle(dep, call_stack, index_in_stack)
                        exit_with_code(
                            1, "Обнаружена циклическая зависимость импортов:\n" + cycle
                        )
                        raise AssertionError("unreachable")

                    if dep_state is None:
                        frames.append((dep, [], 0, False))

            out = list(ir_map.values())
            out = cls._run_project_passes(out)
            out.sort(key=lambda f: str(f.path))
            return out

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")

        except Exception:
            exit_with_code(2, f"Не удалось прочитать файл: {path}")

        raise AssertionError("unreachable")

    @classmethod
    def _build_one(cls, path: Path) -> PyIRFile:
        try:
            source = cls._read_text(path)
            return PyIRBuilder.build_file(source=source, path_to_file=path)

        except SyntaxError as e:
            exit_with_code(1, f"Синтаксическая ошибка\nФайл: {path}\n{e}")

        except Exception as e:
            exit_with_code(3, f"Ошибка при построении IR для {path}: {e}")

        raise AssertionError("unreachable")

    @classmethod
    def _run_file_passes(cls, ir: PyIRFile) -> PyIRFile:
        for p in cls.file_passes:
            try:
                ir = p.run(ir)

            except Exception as e:
                exit_with_code(
                    3, f"Ошибка file-pass {p.__class__.__name__} в {ir.path}: {e}"
                )
                raise AssertionError("unreachable")

        return ir

    @classmethod
    def _run_project_passes(cls, files: list[PyIRFile]) -> list[PyIRFile]:
        for p in cls.project_passes:
            try:
                files = p.run(files)

            except Exception as e:
                exit_with_code(3, f"Ошибка project-pass {p.__class__.__name__}: {e}")
                raise AssertionError("unreachable")

        return files

    @staticmethod
    def _format_cycle(
        dep: Path,
        call_stack: list[Path],
        index_in_stack: dict[Path, int],
    ) -> str:
        start = index_in_stack.get(dep, 0)
        ring = call_stack[start:] + [dep]

        lines: list[str] = []
        for i in range(len(ring) - 1):
            lines.append(f"  {ring[i]} -> {ring[i + 1]}")

        return "\n".join(lines)

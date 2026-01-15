from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

from ..._cli.logging_setup import exit_with_code
from ..py.ir_builder import PyIRBuilder, PyIRFile
from ..py.ir_dataclass import PyIRImport, PyIRImportType


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

                        state[path] = VISITING
                        index_in_stack[path] = len(call_stack)
                        call_stack.append(path)

                        ir = cls._build_one(path)
                        ir = cls._run_file_passes(ir)
                        ir_map[path] = ir

                        deps = cls._collect_and_classify_deps(
                            ir=ir,
                            project_root=root,
                            internal_root=internal_root,
                            current_file=path,
                        )

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

                    if dep_state is None:
                        frames.append((dep, [], 0, False))

            out = list(ir_map.values())
            out = cls._run_project_passes(out)

            out.sort(key=lambda f: str(f.path))
            return out

    # internals
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
            exit_with_code(1, f"Синтаксическая ошибка в файле {path}: {e}")

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

        return ir

    @classmethod
    def _run_project_passes(cls, files: list[PyIRFile]) -> list[PyIRFile]:
        for p in cls.project_passes:
            try:
                files = p.run(files)

            except Exception as e:
                exit_with_code(3, f"Ошибка project-pass {p.__class__.__name__}: {e}")

        return files

    @staticmethod
    def _collect_imports(ir: PyIRFile) -> list[PyIRImport]:
        return [n for n in ir.walk() if isinstance(n, PyIRImport)]

    @classmethod
    def _collect_and_classify_deps(
        cls,
        *,
        ir: PyIRFile,
        project_root: Path,
        internal_root: Path,
        current_file: Path,
    ) -> list[Path]:
        deps: list[Path] = []

        for imp in cls._collect_imports(ir):
            itype, dep_path = cls._classify_import(
                imp=imp,
                project_root=project_root,
                internal_root=internal_root,
                current_file=current_file,
            )
            imp.itype = itype

            if itype == PyIRImportType.EXTERNAL:
                exit_with_code(
                    1,
                    f"Внешний импорт не поддерживается:\n"
                    f"    {'.'.join(imp.modules)}\n"
                    f"Файл: {current_file}",
                )
                raise AssertionError("unreachable")

            if dep_path is not None and itype in (
                PyIRImportType.LOCAL,
                PyIRImportType.INTERNAL,
            ):
                deps.append(dep_path)

        return sorted({p.resolve() for p in deps})

    @classmethod
    def _classify_import(
        cls,
        *,
        imp: PyIRImport,
        project_root: Path,
        internal_root: Path,
        current_file: Path,
    ) -> tuple[PyIRImportType, Path | None]:
        modules = imp.modules or []
        top = modules[0] if modules else None

        if imp.level:
            dep = cls._resolve_relative_to_file(imp, current_file)
            if dep is not None:
                if cls._is_within(dep, project_root):
                    return PyIRImportType.LOCAL, dep

                if cls._is_within(dep, internal_root):
                    return PyIRImportType.INTERNAL, dep

                return PyIRImportType.EXTERNAL, None

            exit_with_code(
                1,
                f"Не удалось разрешить относительный импорт в {current_file}: {modules}",
            )
            raise AssertionError("unreachable")

        dep_local = cls._resolve_absolute_under_base(
            imp, base=project_root, modules=modules
        )
        if dep_local is not None:
            return PyIRImportType.LOCAL, dep_local

        if cls._is_internal_namespace(modules):
            dep_int = cls._resolve_internal_under_root(imp, internal_root)
            if dep_int is not None:
                return PyIRImportType.INTERNAL, dep_int

            return PyIRImportType.INTERNAL, None

        if top is not None and cls._is_stdlib_module(top):
            return PyIRImportType.STD_LIB, None

        return PyIRImportType.EXTERNAL, None

    # resolve helpers
    @classmethod
    def _is_internal_namespace(cls, modules: list[str]) -> bool:
        return len(modules) >= 2 and (modules[0], modules[1]) == cls._INTERNAL_PREFIX

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
    def _resolve_absolute_under_base(
        imp: PyIRImport,
        *,
        base: Path,
        modules: list[str],
    ) -> Path | None:
        if modules:
            p = base.joinpath(*modules).with_suffix(".py")
            if p.exists():
                return p

        if imp.if_from and modules:
            for n in imp.names:
                name = n[0] if isinstance(n, tuple) else n
                p = base.joinpath(*modules, name).with_suffix(".py")
                if p.exists():
                    return p

        return None

    @classmethod
    def _resolve_internal_under_root(
        cls, imp: PyIRImport, internal_root: Path
    ) -> Path | None:
        modules = imp.modules or []
        rest = modules[2:]

        if rest:
            p = internal_root.joinpath(*rest).with_suffix(".py")
            if p.exists():
                return p

        if imp.if_from:
            base = internal_root.joinpath(*rest)
            for n in imp.names:
                name = n[0] if isinstance(n, tuple) else n
                p = base.joinpath(name).with_suffix(".py")
                if p.exists():
                    return p

        return None

    @staticmethod
    def _resolve_relative_to_file(imp: PyIRImport, current_file: Path) -> Path | None:
        base = current_file.parent
        for _ in range(imp.level - 1):
            base = base.parent

        modules = imp.modules or []

        if modules:
            p = base.joinpath(*modules).with_suffix(".py")
            if p.exists():
                return p

        if imp.if_from and modules:
            for n in imp.names:
                name = n[0] if isinstance(n, tuple) else n
                p = base.joinpath(*modules, name).with_suffix(".py")
                if p.exists():
                    return p

        return None

    # cycle formatting
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

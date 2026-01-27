from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterable

from ..._cli.logging_setup import exit_with_code, log_step
from ...config import Py2GluaConfig
from ..py.ir_builder import PyIRBuilder, PyIRFile
from .import_resolver import ImportResolver
from .passes.analysis.symlinks import (
    BuildScopesPass,  # симлинк дата
    CollectDefsPass,  # симлинк дата
    ResolveUsesPass,  # симлинк дата
    RewriteToSymlinksPass,  # симлинк дата
    SymLinkContext,  # симлинк дата
)
from .passes.normalize import (
    AttachDecoratorsPass,
    NormalizeImportsPass,
)


class Compiler:
    normalize_passes: list = [
        NormalizeImportsPass,
        AttachDecoratorsPass,
    ]

    analysis_passes = [
        # === Блок simlinks
        BuildScopesPass,
        CollectDefsPass,
        ResolveUsesPass,
        RewriteToSymlinksPass,
        # ===
    ]

    lowering_passes = []

    project_passes = []

    @classmethod
    def _validate_project(cls):
        root = Py2GluaConfig.source

        if not root.exists():
            exit_with_code(1, f"Папка проекта не найдена: {root}")

        if not root.is_dir():
            exit_with_code(1, f"Указанный путь не является папкой: {root}")

        project_files = sorted(p.resolve() for p in root.rglob("*.py") if p.is_file())

        if not project_files:
            exit_with_code(1, f"В проекте нет ни одного .py файла: {root}")

        try:
            internal_tr = resources.files("py2glua.glua")

        except Exception as e:
            exit_with_code(
                3,
                f"Не удалось найти internal-пакет py2glua.glua: {e}",
            )
            raise AssertionError("unreachable")

        return root, internal_tr, project_files

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True

        except ValueError:
            return False

    @classmethod
    def build_from_src(cls) -> tuple[list[PyIRFile], list[PyIRFile]]:
        root, internal_tr, project_files = cls._validate_project()

        with resources.as_file(internal_tr) as internal_root:
            internal_root = Path(internal_root).resolve()

            resolver = ImportResolver(
                project_root=root,
                internal_root=internal_root,
            )

            VISITING, DONE = 1, 2
            state: dict[Path, int] = {}

            project_ir: dict[Path, PyIRFile] = {}
            internal_ir: dict[Path, PyIRFile] = {}

            index_in_stack: dict[Path, int] = {}
            call_stack: list[Path] = []
            frames: list[tuple[Path, list[Path], int, bool]] = []

            for entry in project_files:
                if entry in state:
                    continue

                frames.append((entry, [], 0, False))

                while frames:
                    path, deps, i, entered = frames.pop()

                    if state.get(path) == DONE:
                        continue

                    if not entered:
                        if state.get(path) == VISITING:
                            exit_with_code(
                                3,
                                f"Сбой обхода зависимостей: {path}",
                            )
                            raise AssertionError("unreachable")

                        state[path] = VISITING
                        index_in_stack[path] = len(call_stack)
                        call_stack.append(path)

                        raw = cls._build_one(path)
                        deps = resolver.collect_deps(
                            ir=raw,
                            current_file=path,
                        )
                        ir = cls._run_normalize(raw)
                        ir.imports = deps.copy()

                        frames.append((path, deps, 0, True))

                        if cls._is_within(path, root):
                            project_ir[path] = ir
                        else:
                            internal_ir[path] = ir

                        continue

                    if i >= len(deps):
                        state[path] = DONE
                        call_stack.pop()
                        index_in_stack.pop(path, None)
                        continue

                    dep = deps[i]
                    frames.append((path, deps, i + 1, True))

                    if state.get(dep) == VISITING:
                        cycle = cls._format_cycle(
                            dep,
                            call_stack,
                            index_in_stack,
                        )
                        exit_with_code(
                            1,
                            "Обнаружена циклическая зависимость импортов:\n" + cycle,
                        )
                        raise AssertionError("unreachable")

                    if dep not in state:
                        frames.append((dep, [], 0, False))

        return (
            list(project_ir.values()),
            list(internal_ir.values()),
        )

    # region passes
    @classmethod
    def _run_normalize(
        cls,
        ir: PyIRFile,
    ) -> PyIRFile:
        for p in cls.normalize_passes:
            try:
                ir = p.run(ir)

            except Exception as e:
                exit_with_code(
                    3,
                    "Ошибка normalize\n"
                    f"Файл: {ir.path}\n"
                    f"Pass: {p.__name__}\n"
                    f"Ошибка: {e}",
                )
                raise AssertionError("unreachable")

        return ir

    @classmethod
    def run_analysis(
        cls,
        project_ir: Iterable[PyIRFile],
        internal_ir: Iterable[PyIRFile],
    ) -> SymLinkContext:
        ctx = SymLinkContext()

        for p in cls.analysis_passes:
            for ir in (*internal_ir, *project_ir):
                try:
                    p.run(ir, ctx)

                except Exception as e:
                    exit_with_code(
                        3,
                        f"Ошибка analysis\n"
                        f"Файл: {ir.path}\n"
                        f"Pass: {p.__name__}\n"
                        f"Ошибка: {e}",
                    )
                    raise AssertionError("unreachable")

        return ctx

    @classmethod
    def run_lowering(
        cls,
        project_ir: Iterable[PyIRFile],
        internal_ir: Iterable[PyIRFile],
        ctx: SymLinkContext,
    ) -> None:
        for p in cls.lowering_passes:
            for ir in (*internal_ir, *project_ir):
                try:
                    p.run(ir, ctx)

                except Exception as e:
                    exit_with_code(
                        3,
                        f"Ошибка lowering\n"
                        f"Файл: {ir.path}\n"
                        f"Pass: {p.__name__}\n"
                        f"Ошибка: {e}",
                    )
                    raise AssertionError("unreachable")

    @classmethod
    def run_project(
        cls,
        project_ir: Iterable[PyIRFile],
        internal_ir: Iterable[PyIRFile],
        ctx: SymLinkContext,
    ) -> None:
        for p in cls.project_passes:
            for ir in (internal_ir, project_ir):
                try:
                    p.run(ir, ctx)

                except Exception as e:
                    exit_with_code(
                        3,
                        f"Ошибка project\nPass: {p.__name__}\nОшибка: {e}",
                    )
                    raise AssertionError("unreachable")

    # endregion

    # public API
    @classmethod
    def build(cls) -> list[PyIRFile]:
        with log_step("[1/5] Подготовка"):
            project_ir, internal_ir = cls.build_from_src()

        with log_step("[2/5] Анализ"):
            ctx = cls.run_analysis(project_ir=project_ir, internal_ir=internal_ir)

        with log_step("[3/5] Сборка"):
            cls.run_lowering(project_ir=project_ir, internal_ir=internal_ir, ctx=ctx)

        with log_step("[4/5] Финализация"):
            cls.run_project(project_ir=project_ir, internal_ir=internal_ir, ctx=ctx)

        project_ir.sort(key=lambda f: str(f.path))
        return project_ir

    # region utils
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
            return PyIRBuilder.build_file(
                source=source,
                path_to_file=path,
            )

        except SyntaxError as e:
            exit_with_code(
                1,
                f"Синтаксическая ошибка\nФайл: {path}\n{e}",
            )

        except Exception as e:
            exit_with_code(
                3,
                f"Ошибка при построении IR для {path}: {e}",
            )

        raise AssertionError("unreachable")

    @staticmethod
    def _format_cycle(
        dep: Path,
        call_stack: list[Path],
        index_in_stack: dict[Path, int],
    ) -> str:
        start = index_in_stack.get(dep, 0)
        ring = call_stack[start:] + [dep]
        return "\n".join(f"  {ring[i]} -> {ring[i + 1]}" for i in range(len(ring) - 1))

    # endregion

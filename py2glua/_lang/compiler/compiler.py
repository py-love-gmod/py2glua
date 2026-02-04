from __future__ import annotations

from importlib import resources
from pathlib import Path

from ..._cli import CompilerExit, log_step
from ...config import Py2GluaConfig
from ..py.ir_builder import PyIRBuilder, PyIRFile
from .import_resolver import ImportResolver
from .passes.analysis import (
    BuildScopesPass,
    CollectDefsPass,
    ResolveUsesPass,
    RewriteToSymlinksPass,
    SymLinkContext,
    TypeFlowPass,
)
from .passes.expand import (
    CollectAnonymousFunctionsPass,
    CollectContextManagersPass,
    CollectInlineFunctionsPass,
    CollectLocalSignaturesPass,
    ExpandContext,
    NormalizeCallArgumentsPass,
    RewriteAnonymousFunctionsPass,
    RewriteClassCtorCallsPass,
    RewriteInlineCallsPass,
    RewriteWithContextManagerPass,
)
from .passes.lowering import (
    CollectGmodApiDeclsPass,
    ConstFoldingPass,
    DcePass,
    FinalizeGmodApiRegistryPass,
    FoldCompileTimeBoolConstsPass,
    NilFoldPass,
    RewriteGmodApiCallsPass,
    StripCompilerDirectiveDefPass,
    StripNoCompileAndGmodApiDefsPass,
    StripPythonOnlyNodesPass,
)
from .passes.normalize import (
    AttachDecoratorsPass,
    NormalizeImportsPass,
)
from .passes.sanity_check import (
    ImportSanityCheckPass,
    RealmDirectiveSanityCheckPass,
    RecursionSanityCheckPass,
    WithSanityCheckPass,
)


class Compiler:
    normalize_passes: list = [
        NormalizeImportsPass,
        AttachDecoratorsPass,
    ]

    sanity_check_passes = [
        ImportSanityCheckPass,  # Запрет import внутри блоков
        WithSanityCheckPass,  # Запрет with блоков с as
        RealmDirectiveSanityCheckPass,  # Проверка корректности __realm__
        RecursionSanityCheckPass,  # Запрещает рекурсию экспанд блоков
    ]

    expand_passes = [
        CollectLocalSignaturesPass,  # args+kwargs нормализация
        NormalizeCallArgumentsPass,  # args+kwargs нормализация
        RewriteClassCtorCallsPass,  # class() -> class.__init__()
        CollectContextManagersPass,  # with развёртка
        RewriteWithContextManagerPass,  # with развёртка
        CollectInlineFunctionsPass,  # inline развёртка
        RewriteInlineCallsPass,  # inline развёртка
        CollectAnonymousFunctionsPass,  # anonymous развёртка
        RewriteAnonymousFunctionsPass,  # anonymous развёртка
    ]

    analysis_passes = [
        # === Блок simlinks
        BuildScopesPass,
        CollectDefsPass,
        ResolveUsesPass,
        RewriteToSymlinksPass,
        # ===
        TypeFlowPass,  # Типизация и проверки типов
    ]

    lowering_passes = [
        NilFoldPass,  # nil fold
        FoldCompileTimeBoolConstsPass,  # DEBUG и TYPE_CHECKING
        StripPythonOnlyNodesPass,  # Стрип анотированных асайнов.
        CollectGmodApiDeclsPass,  # gmod api
        FinalizeGmodApiRegistryPass,  # gmod api
        RewriteGmodApiCallsPass,  # gmod api
        StripNoCompileAndGmodApiDefsPass,  # Стрип no_compile и gmod_api
        StripCompilerDirectiveDefPass,  # Стрип CD
        ConstFoldingPass,  # Конст фолдинг
        DcePass,  # DCE
    ]

    project_passes = []

    # region сборка исходников
    @classmethod
    def _validate_project(cls):
        root = Py2GluaConfig.source

        if not root.exists():
            CompilerExit.user_error(
                f"Папка проекта не найдена: {root}",
                show_path=False,
                show_pos=False,
            )

        if not root.is_dir():
            CompilerExit.user_error(
                f"Указанный путь не является папкой: {root}",
                show_path=False,
                show_pos=False,
            )

        project_files = sorted(p.resolve() for p in root.rglob("*.py") if p.is_file())

        if not project_files:
            CompilerExit.user_error(
                f"В проекте нет ни одного .py файла: {root}",
                show_path=False,
                show_pos=False,
            )

        try:
            internal_tr = resources.files("py2glua.glua")

        except Exception as e:
            CompilerExit.internal_error(
                f"Не удалось найти internal-пакет py2glua.glua: {e}"
            )

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
                            CompilerExit.internal_error(
                                f"Сбой обхода зависимостей: {path}"
                            )
                            raise AssertionError("uneachable")

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
                        CompilerExit.user_error(
                            "Обнаружена циклическая зависимость импортов:\n" + cycle,
                            show_path=False,
                            show_pos=False,
                        )

                    if dep not in state:
                        frames.append((dep, [], 0, False))

        return (
            list(project_ir.values()),
            list(internal_ir.values()),
        )

    # endregion

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
                CompilerExit.internal_error(
                    "Ошибка нормализации\n"
                    f"Файл: {ir.path}\n"
                    f"Pass: {p.__name__}\n"
                    f"Ошибка: {e}",
                )

        return ir

    @classmethod
    def _run_sanity_check(
        cls,
        project_ir: list,
        internal_ir: list,
    ) -> None:
        for p in cls.sanity_check_passes:
            for ir in (*internal_ir, *project_ir):
                try:
                    p.run(ir)

                except Exception as e:
                    CompilerExit.internal_error(
                        "Ошибка валидации\n"
                        f"Файл: {ir.path}\n"
                        f"Pass: {p.__name__}\n"
                        f"Ошибка: {e}",
                    )

    @classmethod
    def _run_expand(
        cls,
        project_ir: list,
        internal_ir: list,
    ) -> None:
        ectx = ExpandContext()

        for p in cls.expand_passes:
            for ir in (*internal_ir, *project_ir):
                try:
                    ir = p.run(ir, ectx)

                except Exception as e:
                    CompilerExit.internal_error(
                        "Ошибка развёртки\n"
                        f"Файл: {ir.path}\n"
                        f"Pass: {p.__name__}\n"
                        f"Ошибка: {e}",
                    )

    @classmethod
    def _run_analysis(
        cls,
        project_ir: list,
        internal_ir: list,
    ) -> SymLinkContext:
        slctx = SymLinkContext()

        for p in cls.analysis_passes:
            for ir in (*internal_ir, *project_ir):
                try:
                    p.run(ir, slctx)

                except Exception as e:
                    CompilerExit.internal_error(
                        f"Ошибка анализа\n"
                        f"Файл: {ir.path}\n"
                        f"Pass: {p.__name__}\n"
                        f"Ошибка: {e}",
                    )

        return slctx

    @classmethod
    def _run_lowering(
        cls, project_ir: list, internal_ir: list, slctx: SymLinkContext
    ) -> None:
        for p in cls.lowering_passes:
            for ir in (*internal_ir, *project_ir):
                try:
                    p.run(ir, slctx)

                except Exception as e:
                    CompilerExit.internal_error(
                        "Ошибка упрощения\n"
                        f"Файл: {ir.path}\n"
                        f"Pass: {p.__name__}\n"
                        f"Ошибка: {e}",
                    )

    # endregion

    # public API
    @classmethod
    def build(cls) -> list[PyIRFile]:
        with log_step("[1/7] Сборка исходников..."):
            project_ir, internal_ir = cls.build_from_src()

        with log_step("[2/7] Проверка валидности..."):
            cls._run_sanity_check(project_ir, internal_ir)

        with log_step("[3/7] Развёртка..."):
            cls._run_expand(project_ir, internal_ir)

        with log_step("[4/7] Анализ..."):
            slctx = cls._run_analysis(project_ir, internal_ir)

        with log_step("[5/7] Упрощение..."):
            cls._run_lowering(project_ir, internal_ir, slctx)

        with log_step("[6/7] Сборка..."):
            pass

        project_ir.sort(key=lambda f: str(f.path))
        return project_ir

    # region utils
    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")

        except Exception:
            CompilerExit.system_error(f"Не удалось прочитать файл: {path}")

    @classmethod
    def _build_one(cls, path: Path) -> PyIRFile:
        try:
            source = cls._read_text(path)
            return PyIRBuilder.build_file(
                source=source,
                path_to_file=path,
            )

        except SyntaxError as e:
            CompilerExit.user_error(
                f"Синтаксическая ошибка\n{e}",
                path=path,
                show_pos=False,
            )

        except Exception as e:
            CompilerExit.internal_error(
                f"Ошибка при построении IR для {path}: {e}",
            )

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

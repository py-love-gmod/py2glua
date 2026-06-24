from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from _utils import Config, Shutdown, logger
from plg_reader import build_python_files_dir

from .dgc import DependencyGraphConstructor
from .dt import CompilationUnit
from .import_work import ImportCollector, ImportResolver
from .symbol_table import SymlinkResolver
from .transforms import CallNormalizer, ConstructorResolver, RealmCollector

SUPPORTED_IMPORTS = frozenset(
    {
        "typing",
        "enum",
    }
)


class Compiler:
    _TRANSFORMS: list[Callable[[CompilationUnit], CompilationUnit]] = [
        RealmCollector.collect,  # Сборка и обработка __realm__
        ConstructorResolver.resolve,  # Class() -> Class.__init__()
        CallNormalizer.normalize,  # kwargs -> args
    ]

    @classmethod
    def _preparation(cls, input_path: Path, std_dir: Path) -> CompilationUnit:
        try:
            irs = build_python_files_dir(input_path)

        except Exception as e:
            Shutdown.user_error(str(e))

        irs = ImportCollector.collect(
            irs,
            search_roots=[input_path],
            builtin_modules=SUPPORTED_IMPORTS,
            prefix_roots={"py2glua.std": std_dir},
        )
        irs = ImportResolver.resolve_imports(irs, builtin_modules=SUPPORTED_IMPORTS)

        graph = DependencyGraphConstructor.build(irs)
        irs, module_scopes, node_to_scope = SymlinkResolver.resolve(irs)
        unit = CompilationUnit(irs, module_scopes, node_to_scope, graph, {})
        return unit

    @classmethod
    def _apply_transforms(cls, unit: CompilationUnit) -> CompilationUnit:
        for transformer in cls._TRANSFORMS:
            unit = transformer(unit)

        return unit

    @classmethod
    def compile(cls) -> None:
        input_path, output_path, p2g_root_dir = (  # noqa: F841
            Config.get_path_input(),
            Config.get_path_output(),
            Config.get_plg_root_path(),
        )
        logger.info("Компиляция запущена, ожидайте...")
        t0 = perf_counter()

        unit = cls._preparation(input_path, p2g_root_dir / "std")
        unit = cls._apply_transforms(unit)

        dt = perf_counter() - t0
        logger.info("Компиляция выполнена за %.3fs", dt)
        logger.info(
            "Но файлы аутпута спиздили гомогномики. https://www.youtube.com/watch?v=j3hOd7u35no"
        )

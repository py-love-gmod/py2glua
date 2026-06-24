from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from plg_reader import build_python_files_dir
from _utils import Config, Shutdown, logger

from .dgc import DependencyGraphConstructor
from .dt import CompilationUnit
from .import_work import ImportCollector, ImportResolver
from .symbol_table import SymlinkResolver
from .transforms import CallNormalizer, ConstructorResolver

SUPPORTED_IMPORTS = frozenset(
    {
        "py2glua",
        #
        "typing",
        "enum",
    }
)


class Compiler:
    _TRANSFORMS: list[Callable[[CompilationUnit], CompilationUnit]] = [
        ConstructorResolver.resolve,  # Class() -> Class.__init__()
        CallNormalizer.normalize,  # kwargs -> args
    ]

    @classmethod
    def _preparation(cls, input_path: Path, py2glua_root: Path) -> CompilationUnit:
        try:
            irs = build_python_files_dir(input_path)  # Парсинг и сборка в IR

        except Exception as e:
            Shutdown.user_error(str(e))

        # Дособираем необходимые модули
        irs = ImportCollector.collect(irs, search_roots=[input_path, py2glua_root])
        # Проверяем что валидно всё
        irs = ImportResolver.resolve_imports(irs, builtin_modules=SUPPORTED_IMPORTS)

        graph = DependencyGraphConstructor.build(irs)  # Граф зависимости

        # Создание сим таблицы
        irs, module_scopes, node_to_scope = SymlinkResolver.resolve(irs)
        unit = CompilationUnit(irs, module_scopes, node_to_scope, graph)

        return unit

    @classmethod
    def _apply_transforms(cls, unit: CompilationUnit) -> CompilationUnit:
        for transformer in cls._TRANSFORMS:
            unit = transformer(unit)

        return unit

    @classmethod
    def compile(cls) -> None:
        input_path, output_path, py2glua_root = (  # noqa: F841
            Config.get_path_input(),
            Config.get_path_output(),
            Config.get_path_plg(),
        )
        logger.info("Компиляция запущена, ожидайте...")
        t0 = perf_counter()

        unit = cls._preparation(input_path, py2glua_root)
        unit = cls._apply_transforms(unit)

        dt = perf_counter() - t0
        logger.info("Компиляция выполнена за %.3fs", dt)
        logger.info(
            "Но файлы аутпута спиздили гомогномики. https://www.youtube.com/watch?v=j3hOd7u35no"
        )

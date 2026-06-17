from pathlib import Path

from plg_reader import build_python_files_dir
from utils import Config, Shutdown

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

        unit = ConstructorResolver.resolve(unit)  # Class() -> Class.__init__()
        unit = CallNormalizer.normalize(unit)  # kwargs -> args

        return unit

    @classmethod
    def compile(cls) -> None:
        input_path = Config.get_path_input()
        output_path = Config.get_path_output()
        py2glua_root = Config.get_path_plg()

        unit = cls._preparation(input_path, py2glua_root)

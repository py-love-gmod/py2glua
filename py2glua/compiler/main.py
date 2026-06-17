from pathlib import Path

from plg_reader import IRFile, build_python_files_dir
from utils import Config, Shutdown

from ._dgc import DependencyGraphConstructor
from .import_work import ImportCollector, ImportResolver
from .lua_dumper import LuaDumper

SUPPORTED_BUILTINS = frozenset({})


class Compiler:
    @classmethod
    def _build_irs(cls, input_path: Path) -> dict[str, IRFile]:
        try:
            return build_python_files_dir(input_path)

        except Exception as e:
            Shutdown.user_error(str(e))

    @classmethod
    def compile(cls) -> None:
        input_path = Config.get_path_input()
        output_path = Config.get_path_output()
        py2glua_root = Config.get_path_plg()

        irs = cls._build_irs(input_path)

        irs = ImportCollector.collect(irs, search_roots=[input_path, py2glua_root])

        irs = ImportResolver.resolve_imports(irs, builtin_modules=SUPPORTED_BUILTINS)

        graph = DependencyGraphConstructor.build(irs)  # noqa: F841

        LuaDumper.dump(irs, output_path)

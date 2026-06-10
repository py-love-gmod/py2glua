from pathlib import Path

from plg_reader import IRFile, build_python_files_dir
from utils import Config, Shutdown

from ._dgc import DependencyGraphConstructor
from ._import_resolver import ImportResolver
from .lua_dumper import LuaDumper


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

        #
        irs = cls._build_irs(input_path)

        #
        irs = ImportResolver.resolve_imports(irs)
        graph = DependencyGraphConstructor.build(irs)  # noqa: F841

        #
        LuaDumper.dump(irs, output_path)

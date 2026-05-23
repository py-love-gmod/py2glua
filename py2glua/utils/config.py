from argparse import Namespace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


class Config:
    cli_args: Namespace = Namespace()
    _data: dict[str, Any] = {}

    @staticmethod
    def version() -> str:
        try:
            return version("py2glua")

        except PackageNotFoundError:
            return "0.0.0dev"

    @classmethod
    def path_setup(cls, input_path: Path, output_path: Path, plg_path: Path) -> None:
        cls._data["path_in"] = input_path
        cls._data["path_out"] = output_path
        cls._data["path_plg"] = plg_path

    @classmethod
    def namespace_setup(cls, namescape: str | None) -> None:
        # TODO Сделать валидацию ns и рандомизацию в случае None
        cls._data["ns"] = namescape

    # region getters
    @classmethod
    def get_path_input(cls) -> Path:
        return cls._data["path_in"]

    @classmethod
    def get_path_output(cls) -> Path:
        return cls._data["path_out"]

    @classmethod
    def get_path_plg(cls) -> Path:
        return cls._data["path_plg"]

    @classmethod
    def get_namespace(cls) -> str:
        return cls._data["ns"]

    # endregion

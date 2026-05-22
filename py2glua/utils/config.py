from argparse import Namespace
from importlib.metadata import PackageNotFoundError, version


class Config:
    cli_args: Namespace = Namespace()

    @staticmethod
    def version() -> str:
        try:
            return version("py2glua")

        except PackageNotFoundError:
            return "0.0.0dev"

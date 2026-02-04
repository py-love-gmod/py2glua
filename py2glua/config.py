from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


class Py2GluaConfig:
    source: Path = Path()
    output: Path = Path()

    verbose: bool = False
    debug: bool = False

    namespace: str = ""

    @classmethod
    def version(cls) -> str:
        try:
            return version("py2glua")

        except PackageNotFoundError:
            return "0.0.0dev"

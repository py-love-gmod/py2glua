from pathlib import Path

from plg_reader import IRFile
from utils import logger


class LuaDumper:
    @classmethod
    def dump(cls, irs: dict[str, IRFile], output_path: Path) -> None:
        for path, ir in irs.items():
            logger.debug(f"{path=}\n{ir.pretty()}")

import logging
from pathlib import Path

logger = logging.getLogger("py2glua")


class Compiler:
    @classmethod
    def rgrob_files(cls, source: Path) -> list[Path] | None:
        if not source.exists():
            logger.warning(f"[compiler] Путь {source} не существует")
            return None

        if not source.is_dir():
            logger.warning(f"[compiler] Путь {source} не является папкой")
            return None

        list_of_files: list[Path] = []
        for path in source.rglob(".py"):
            list_of_files.append(path)

        return list_of_files

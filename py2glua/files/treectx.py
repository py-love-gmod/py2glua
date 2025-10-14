import logging
from pathlib import Path

from .filectx import FileCTX

logger = logging.getLogger("py2glua")


class TreeCTX:
    def __init__(self) -> None:
        self.files_ctxs: dict[str, FileCTX] = {}

    def add_file(self, file_ctx: FileCTX) -> bool:
        file_path = str(file_ctx.file_path.resolve())
        if file_path in self.files_ctxs:
            return False

        self.files_ctxs[file_path] = file_ctx
        return True

    def remove_file(self, path: Path) -> bool:
        fp = str(path.resolve())
        if fp in self.files_ctxs:
            del self.files_ctxs[fp]
            return True

        return False

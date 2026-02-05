from __future__ import annotations

from typing import Final

from ....py.ir_dataclass import (
    PyIRComment,
    PyIRFile,
    PyIRImport,
    PyIRNode,
)
from ..analysis.symlinks import SymLinkContext


class CleanUpEmptyFilesPass:
    """
    Project-pass.

    Removes PyIRFile objects that do not contribute any runtime or
    semantic value to codegen.

    A file is considered empty if its top-level body contains only:
      - PyIRImport
      - PyIRComment
    or if body is empty.
    """

    _ALLOWED_EMPTY_NODES: Final[tuple[type[PyIRNode], ...]] = (
        PyIRImport,
        PyIRComment,
    )

    @staticmethod
    def run(
        files: list[PyIRFile],
        ctx: SymLinkContext,
    ) -> list[PyIRFile]:
        _ = ctx

        out: list[PyIRFile] = []

        for ir in files:
            if not CleanUpEmptyFilesPass._is_effectively_empty(ir):
                out.append(ir)

        return out

    @staticmethod
    def _is_effectively_empty(ir: PyIRFile) -> bool:
        for node in ir.body:
            if isinstance(node, CleanUpEmptyFilesPass._ALLOWED_EMPTY_NODES):
                continue

            return False

        return True

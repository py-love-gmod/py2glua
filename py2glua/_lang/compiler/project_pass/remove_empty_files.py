from __future__ import annotations

from ...py.ir_dataclass import PyIRComment, PyIRFile, PyIRImport


class RemoveEmptyFilesPass:
    @classmethod
    def run(cls, files: list[PyIRFile]) -> list[PyIRFile]:
        out: list[PyIRFile] = []

        for f in files:
            if cls._is_non_empty(f):
                out.append(f)

        return out

    @staticmethod
    def _is_non_empty(file: PyIRFile) -> bool:
        for node in file.body:
            if not isinstance(node, (PyIRImport, PyIRComment)):
                return True

        return False

from pathlib import Path


class CompileError(Exception):
    def __init__(self, msg: str, file_path: Path, lineno: int, col_offset: int):
        self.msg = msg
        self.file_path = file_path
        self.lineno = lineno
        self.col_offset = col_offset
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            f"{self.msg}\n"
            f"PATH: {self.file_path}\n"
            f"LINE|OFFSET: {self.lineno}|{self.col_offset}"
        )


__all__ = ["CompileError"]

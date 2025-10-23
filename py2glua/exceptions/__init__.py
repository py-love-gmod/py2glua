from pathlib import Path


class CompileError(Exception):
    """Base compiler error that includes source location details."""

    def __init__(
        self,
        msg: str,
        file_path: Path | None,
        lineno: int | None,
        col_offset: int | None,
    ):
        self.msg = msg
        self.file_path = file_path or Path("<unknown>")
        self.lineno = lineno if lineno is not None else -1
        self.col_offset = col_offset if col_offset is not None else -1
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            f"{self.msg}\n"
            f"PATH: {self.file_path}\n"
            f"LINE|OFFSET: {self.lineno}|{self.col_offset}"
        )


class DeliberatelyUnsupportedError(CompileError):
    """Raised when encountering a feature that is intentionally not supported."""


__all__ = ["CompileError", "DeliberatelyUnsupportedError"]

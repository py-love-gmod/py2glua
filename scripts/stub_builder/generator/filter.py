from pathlib import Path
from typing import Set


class FunctionFilter:
    _nuked_functions: Set[str] = set()
    _initialized: bool = False

    @classmethod
    def init(cls, nuke_file_path: Path) -> None:
        if cls._initialized:
            return

        nuke_file_path = nuke_file_path / "nuke.txt"
        if nuke_file_path.exists():
            with open(nuke_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        cls._nuked_functions.add(line)

        cls._initialized = True

    @classmethod
    def is_nuked(cls, method_name: str, class_name: str | None = None) -> bool:
        if class_name:
            full_path = f"{class_name}.{method_name}"
            if full_path in cls._nuked_functions:
                return True

        return method_name in cls._nuked_functions

from pathlib import Path


class FunctionFilter:
    def __init__(self, nuke_file: Path | None = None):
        self.excluded: set[str] = set()
        if nuke_file:
            self.load(nuke_file)

    def load(self, path: Path) -> None:
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self.excluded.add(line)

    def is_excluded(self, method_name: str, class_name: str | None = None) -> bool:
        if class_name:
            full = f"{class_name}.{method_name}"
            if full in self.excluded:
                return True

        return method_name in self.excluded

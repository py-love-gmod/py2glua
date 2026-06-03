from pathlib import Path
from typing import Optional


class TypeOverride:
    def __init__(self, rules_path: Optional[Path] = None):
        self.global_rules: dict[str, str] = {}
        self.local_rules: dict[str, str] = {}
        if rules_path:
            self.load(rules_path)

    def load(self, path: Path) -> None:
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#") or "->" not in line:
                    continue

                try:
                    left, right = line.split("->", 1)

                except ValueError:
                    print(f"[TypeOverride] line {line_no}: no '->', skipping")
                    continue

                left = left.strip()
                right = right.strip()
                if not left or not right:
                    print(f"[TypeOverride] line {line_no}: empty side, skipping")
                    continue

                if "." in left:
                    self.local_rules[left] = right

                else:
                    self.global_rules[left.lower()] = right

    def get(self, original_type: str, context: str | None = None) -> str:
        if "|" in original_type:
            parts = [p.strip() for p in original_type.split("|")]
            resolved = [self.get(part, context) for part in parts]
            return " | ".join(resolved)

        if context and context in self.local_rules:
            return self.local_rules[context]

        if original_type.lower() in self.global_rules:
            return self.global_rules[original_type.lower()]

        return original_type

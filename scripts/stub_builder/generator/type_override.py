from pathlib import Path
from typing import Dict


class TypeOverride:
    _global_rules: Dict[str, str] = {}
    _local_rules: Dict[str, str] = {}
    _initialized: bool = False
    _rules_file_path: Path | None = None

    @classmethod
    def init(cls, lookup_paths: Path) -> None:
        if cls._initialized:
            return

        cls._rules_file_path = cls._find_rules_file(lookup_paths)

        if cls._rules_file_path:
            cls._load_from_file()

        cls._initialized = True

    @classmethod
    def reload(cls) -> None:
        if cls._rules_file_path and cls._rules_file_path.exists():
            cls._global_rules.clear()
            cls._local_rules.clear()
            cls._load_from_file()

    @classmethod
    def get(cls, original_type: str, context: str | None = None) -> str:
        normalized_type = original_type.lower()

        if context:
            local_key = cls._normalize_local_key(context)
            if local_key in cls._local_rules:
                return cls._local_rules[local_key]

        if normalized_type in cls._global_rules:
            return cls._global_rules[normalized_type]

        return original_type

    @classmethod
    def _find_rules_file(cls, lookup_path: Path) -> Path | None:
        normal_file = lookup_path / "type_overrides.txt"
        if normal_file.exists() and normal_file.is_file():
            return normal_file

        return None

    @classmethod
    def _load_from_file(cls) -> None:
        if not cls._rules_file_path:
            return

        try:
            with open(cls._rules_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    if not line or line.startswith("#"):
                        continue

                    if "->" not in line:
                        print(
                            f"[TypeOverride] Warning: line {line_num}: no '->' separator, skipping"
                        )
                        continue

                    left, right = line.split("->", 1)
                    left = left.strip()
                    right = right.strip()

                    if not left or not right:
                        print(
                            f"[TypeOverride] Warning: line {line_num}: empty left or right side, skipping"
                        )
                        continue

                    if (
                        "." in left
                        or left.endswith(".return")
                        or "return" in left.split(".")
                    ):
                        cls._local_rules[left] = right

                    else:
                        cls._global_rules[left.lower()] = right

        except Exception as e:
            print(f"[TypeOverride] Error loading rules: {e}")

    @classmethod
    def _normalize_local_key(cls, context: str) -> str:
        if "." in context and ":" not in context:
            parts = context.split(".", 1)
            if len(parts) == 2:
                return f"{parts[0]}:{parts[1]}"

        return context

    @classmethod
    def reset(cls) -> None:
        cls._global_rules.clear()
        cls._local_rules.clear()
        cls._initialized = False
        cls._rules_file_path = None

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized


def override_type(original_type: str, context: str | None = None) -> str:
    return TypeOverride.get(original_type, context)

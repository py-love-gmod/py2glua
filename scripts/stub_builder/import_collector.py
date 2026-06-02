from collections import defaultdict
from typing import Dict, List, Set


class ImportCollector:
    _types: Set[str] = set()
    _current_file: str | None = None

    TYPE_TO_IMPORT: Dict[str, str] = {
        "Callable": "collections.abc",
        "Sequence": "typing",
        "Iterator": "collections.abc",
        "Iterable": "collections.abc",
        "Any": "typing",
        "Enum": "enum",
        "dataclass": "dataclasses",
    }

    @classmethod
    def start_file(cls, filename: str) -> None:
        """Начинает новый файл, сбрасывает типы."""
        cls._types.clear()
        cls._current_file = filename

    @classmethod
    def reset(cls) -> None:
        cls._types.clear()

    @classmethod
    def add_type(cls, type_str: str) -> None:
        if not type_str or type_str in [
            "str",
            "int",
            "float",
            "bool",
            "None",
        ]:
            return

        clean_type = cls._clean_type(type_str)

        if clean_type in cls.TYPE_TO_IMPORT:
            cls._types.add(clean_type)

    @classmethod
    def _clean_type(cls, type_str: str) -> str:
        if type_str == "*args":
            return "Any"

        if "[" in type_str and type_str.endswith("]"):
            base = type_str[: type_str.index("[")]
            return base

        if " | " in type_str:
            return "Union"

        return type_str

    @classmethod
    def get_imports(cls) -> List[str]:
        if not cls._types:
            return []

        imports_by_module: Dict[str, Set[str]] = defaultdict(set)

        for type_name in cls._types:
            module = cls.TYPE_TO_IMPORT.get(type_name)
            if module:
                imports_by_module[module].add(type_name)

        lines = ["from __future__ import annotations", ""]

        for module, types in sorted(imports_by_module.items()):
            lines.append(f"from {module} import {', '.join(sorted(types))}")

        if len(lines) > 2:
            lines.append("")

        return lines

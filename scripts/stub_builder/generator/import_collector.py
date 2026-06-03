from typing import FrozenSet

IMPORT_MAP: dict[str, str] = {
    "dataclass": "from dataclasses import dataclass",
    "Callable": "from collections.abc import Callable",
    "Enum": "from enum import Enum",
    "Any": "from typing import Any",
    "deprecated": "from warnings import deprecated",
}

BUILTINS: FrozenSet[str] = frozenset(
    {"str", "int", "float", "bool", "list", "dict", "tuple", "None"}
)


def resolve_standard_imports(used_types: set[str]) -> list[str]:
    imports: set[str] = set()
    for t in used_types:
        if t in BUILTINS or t in ("tuple", "None"):
            continue

        if t in IMPORT_MAP:
            imports.add(IMPORT_MAP[t])

        else:
            pass

    return sorted(imports)

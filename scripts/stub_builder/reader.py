import json
import zipfile
from pathlib import Path
from typing import Any, List, Tuple


def load_objects_from_zip(
    zip_path: Path,
    ignore_names: set | None = None,
) -> List[Tuple[str, Any]]:
    if ignore_names is None:
        ignore_names = {"__metadata.json"}

    results = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if Path(name).name in ignore_names:
                continue

            if not name.lower().endswith(".json"):
                continue

            try:
                data = json.loads(zf.read(name).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            results.append((name, data))

    return results

from collections import defaultdict
from typing import Any, Dict, List, Tuple


def extract_obj(data):
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
        return None
    return data if isinstance(data, dict) else None


def build_full_schema(entries: List[Tuple[str, Any]]) -> Dict[str, dict]:
    schema = defaultdict(
        lambda: {
            "kind": None,
            "description": "",
            "parent": None,
            "url": "",
            "methods": [],
            "fields": [],
            "items": [],
        }
    )

    for path, data in entries:
        obj = extract_obj(data)
        if not obj:
            continue
        t = obj.get("type")

        if t in ("library", "class", "panel", "struct", "enum"):
            name = obj["name"]
            schema[name]["kind"] = t
            schema[name]["description"] = obj.get("description", "")
            schema[name]["url"] = obj.get("url", "")
            if t in ("class", "panel"):
                schema[name]["parent"] = obj.get("parent")

            elif t == "struct":
                schema[name]["fields"] = obj.get("fields", [])

            elif t == "enum":
                schema[name]["items"] = obj.get("items", [])

        elif t == "hook":
            name = obj["name"]
            schema[name]["kind"] = "hook"
            schema[name]["description"] = obj.get("description", "")
            schema[name]["parent"] = obj.get("parent")
            schema[name]["arguments"] = obj.get("arguments", [])
            schema[name]["returns"] = obj.get("returns", [])

        elif t in ("libraryfunc", "classfunc", "panelfunc"):
            parent = obj["parent"]
            if schema[parent]["kind"] is None:
                guessed = t.replace("func", "")
                schema[parent]["kind"] = guessed

            schema[parent]["methods"].append(obj)

    return dict(schema)

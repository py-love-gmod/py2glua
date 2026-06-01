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
    # schema: имя_сущности -> { kind, description, parent, methods, ... }
    schema = defaultdict(
        lambda: {"kind": None, "description": "", "parent": None, "methods": []}
    )

    for path, data in entries:
        obj = extract_obj(data)
        if not obj:
            continue
        t = obj.get("type")
        if t == "library":
            name = obj["name"]
            schema[name]["kind"] = "library"
            schema[name]["description"] = obj.get("description", "")
            schema[name]["url"] = obj.get("url", "")
        elif t == "libraryfunc":
            parent = obj["parent"]
            # гарантируем, что родитель существует в schema
            if schema[parent]["kind"] is None:
                schema[parent]["kind"] = "library"
            schema[parent]["methods"].append(obj)
        elif t == "classfunc":
            parent = obj["parent"]
            if schema[parent]["kind"] is None:
                schema[parent]["kind"] = "class"
            schema[parent]["methods"].append(obj)
        elif t == "class":
            name = obj["name"]
            schema[name]["kind"] = "class"
            schema[name]["description"] = obj.get("description", "")
            schema[name]["parent"] = obj.get("parent")
        elif t == "panel":
            name = obj["name"]
            schema[name]["kind"] = "panel"
            schema[name]["description"] = obj.get("description", "")
            schema[name]["parent"] = obj.get("parent")
        elif t == "panelfunc":
            parent = obj["parent"]
            if schema[parent]["kind"] is None:
                schema[parent]["kind"] = "panel"
            schema[parent]["methods"].append(obj)
        elif t == "struct":
            name = obj["name"]
            schema[name]["kind"] = "struct"
            schema[name]["description"] = obj.get("description", "")
            schema[name]["fields"] = obj.get("fields", [])
        elif t == "hook":
            name = obj["name"]
            schema[name]["kind"] = "hook"
            schema[name]["description"] = obj.get("description", "")
            schema[name]["arguments"] = obj.get("arguments", [])
            schema[name]["returns"] = obj.get("returns", [])
        elif t == "enum":
            name = obj["name"]
            schema[name]["kind"] = "enum"
            schema[name]["description"] = obj.get("description", "")
            schema[name]["items"] = obj.get("items", [])

    return dict(schema)

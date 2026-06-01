from .utils import format_docstring


def generate(name: str, data: dict) -> str:
    lines = ["from enum import Enum", "", ""]
    class_name = name
    lines.append(f"class {class_name}(Enum):")
    desc = data.get("description", "")
    if desc:
        lines.append(format_docstring(desc, 4))

    else:
        lines.append('    """..."""')

    for item in data.get("items", []):
        key = item.get("key", "")
        value = item.get("value", "")
        item_desc = item.get("description", "")
        if item_desc:
            lines.append(f"    {key} = {value}  # {item_desc}")

        else:
            lines.append(f"    {key} = {value}")

    lines.append("")
    return "\n".join(lines)

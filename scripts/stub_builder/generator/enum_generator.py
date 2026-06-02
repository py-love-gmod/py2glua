from .utils import format_docstring


def generate_enum(name: str, data: dict) -> str:
    lines = [
        "from enum import Enum",
        "",
        "",
    ]
    lines.append(f"class {name}(Enum):")

    desc = data.get("description", "")
    if not desc:
        desc = "..."

    lines.append(format_docstring(desc, 4))
    lines.append("")

    for item in data.get("items", []):
        key = item.get("key", "")
        value = item.get("value", "")
        lines.append(f"    {key} = {value}")

        if desc := item.get("description", ""):
            lines.append(format_docstring(desc, 4))

        lines.append("")

    lines.append("")
    return "\n".join(lines)

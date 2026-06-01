from .utils import format_docstring


def generate(name: str, data: dict) -> str:
    lines = [
        "from __future__ import annotations",
        "from dataclasses import dataclass",
        "",
        "",
    ]
    class_name = name
    lines.append("@dataclass")
    lines.append(f"class {class_name}:")
    desc = data.get("description", "")
    if desc:
        lines.append(format_docstring(desc, 4))

    else:
        lines.append('    """..."""')

    for field in data.get("fields", []):
        fname = field.get("name", "")
        ftype = field.get("type", "Any")
        default = field.get("default")
        if default is not None:
            lines.append(f"    {fname}: {ftype} = {default}")

        else:
            lines.append(f"    {fname}: {ftype}")

    lines.append("")
    return "\n".join(lines)

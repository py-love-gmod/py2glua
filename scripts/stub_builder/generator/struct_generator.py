from .type_override import TypeOverride
from .utils import format_docstring


def generate_struct(name: str, data: dict) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "",
    ]
    lines.append("@dataclass")
    lines.append(f"class {name}:")

    desc = data.get("description", "")
    if not desc:
        desc = "..."

    lines.append(format_docstring(desc, 4))
    lines.append("")

    for field in data.get("fields", []):
        field: dict

        fname = field.get("name", "")
        ftype = field.get("type", "Any")

        context = f"{name}.{fname}"
        ftype = TypeOverride.get(ftype, context)

        default = field.get("default")
        if default is not None:
            lines.append(f"    {fname}: {ftype} = {default}")

        else:
            lines.append(f"    {fname}: {ftype}")

        if desc := field.get("description"):
            lines.append(format_docstring(desc, 4))

        lines.append("")

    lines.append("")
    return "\n".join(lines)

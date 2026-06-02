from .function_generator import generate_function
from .utils import format_docstring


def generate_class(name: str, data: dict, is_static: bool = False) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "",
    ]
    parent = data.get("parent")
    if parent:
        lines.append(f"class {name}({parent}):")

    else:
        lines.append(f"class {name}:")

    desc = data.get("description", "")
    if desc:
        lines.append(format_docstring(desc, 4))

    else:
        lines.append('    """..."""')

    lines.append("")

    for method in data.get("methods", []):
        lines.append(generate_function(method, is_static=is_static))
        lines.append("")

    return "\n".join(lines)

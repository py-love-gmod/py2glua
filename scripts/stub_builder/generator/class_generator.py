from .function_generator import generate_function
from .utils import format_docstring


def generate_class(
    name: str,
    data: dict,
    is_static: bool = False,
) -> tuple[str, set[str]]:
    all_types = set()

    lines = []
    parent = data.get("parent")
    if parent:
        lines.append(f"class {name}({parent}):")

    else:
        lines.append(f"class {name}:")

    lines.append(format_docstring(data.get("description", "..."), 4))
    lines.append("")

    for method in data.get("methods", []):
        func_code, used_types = generate_function(
            method, is_static=is_static, class_name=name
        )
        if func_code:
            lines.append(func_code)
            lines.append("")
            all_types.update(used_types)

    return "\n".join(lines), all_types

from ..models import EnumDef
from .base import GeneratedCode
from .utils import format_docstring


def generate_enum(enum_def: EnumDef) -> GeneratedCode:
    lines = [
        f"class {enum_def.name}(Enum):",
        format_docstring(enum_def.description, 4),
        "",
    ]
    for item in enum_def.items:
        lines.append(f"    {item.key} = {item.value}")
        if item.description:
            lines.append(format_docstring(item.description, 4))

        lines.append("")

    return GeneratedCode("\n".join(lines), {"Enum"})

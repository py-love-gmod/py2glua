from ..models import StructDef
from ..type_override import TypeOverride
from .base import GeneratedCode
from .utils import format_docstring


def generate_struct(
    struct_def: StructDef,
    type_override: TypeOverride,
) -> GeneratedCode:
    used_types: set[str] = {"dataclass"}
    lines = ["@dataclass", f"class {struct_def.name}:"]
    lines.append(format_docstring(struct_def.description, 4))
    lines.append("")

    for field in struct_def.fields:
        context = f"{struct_def.name}.{field.name}"
        ftype = type_override.get(field.type, context)
        if ftype not in ("Any", "None", "vararg", "*args", "**kwargs"):
            used_types.add(ftype)

        default = getattr(field, "default", None)

        if default is not None:
            lines.append(f"    {field.name}: {ftype} = {default}")

        else:
            lines.append(f"    {field.name}: {ftype}")

        if field.description:
            lines.append(format_docstring(field.description, 4))

        lines.append("")

    return GeneratedCode("\n".join(lines), used_types)

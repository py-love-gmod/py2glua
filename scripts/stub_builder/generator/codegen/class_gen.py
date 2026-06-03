from ..function_filter import FunctionFilter
from ..models import ClassDef
from ..type_override import TypeOverride
from .base import GeneratedCode
from .function import generate_function
from .utils import format_docstring


def generate_class(
    class_def: ClassDef,
    type_override: TypeOverride,
    function_filter: FunctionFilter,
) -> GeneratedCode:
    lines: list[str] = []
    if class_def.parent:
        lines.append(f"class {class_def.name}({class_def.parent}):")

    else:
        lines.append(f"class {class_def.name}:")

    lines.append(format_docstring(class_def.description, 4))
    lines.append("")

    used_types: set[str] = set()
    for method in class_def.methods:
        method.is_static = class_def.is_static
        gen = generate_function(
            method, type_override, function_filter, class_name=class_def.name
        )
        if gen is not None:
            lines.append(gen.code)
            lines.append("")
            used_types.update(gen.used_types)

    return GeneratedCode("\n".join(lines), used_types)

from __future__ import annotations

from .class_builder import render_classes
from .function_builder import render_functions
from .models import ApiClassSpec, ApiFunctionSpec, ApiTypeVarSpec
from .type_var_builder import render_type_vars


def render_api_module(
    *,
    imports: list[str],
    type_vars: list[ApiTypeVarSpec],
    functions: list[ApiFunctionSpec],
    classes: list[ApiClassSpec],
) -> str:
    out: list[str] = ["from __future__ import annotations"]

    if imports or type_vars or functions or classes:
        out.append("")

    for imp in imports:
        out.append(imp)

    tv_lines = render_type_vars(type_vars, indent=0)

    if imports and (tv_lines or functions or classes):
        out.append("")

    if tv_lines:
        out.extend(tv_lines)
        if functions or classes:
            out.append("")

    fn_lines = render_functions(functions, indent=0)
    cls_lines = render_classes(classes, indent=0)

    if fn_lines:
        out.extend(fn_lines)
        if cls_lines:
            out.append("")

    if cls_lines:
        out.extend(cls_lines)

    return "\n".join(out).rstrip() + "\n"

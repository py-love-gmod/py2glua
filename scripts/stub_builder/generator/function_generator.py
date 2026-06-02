from .arg_parser import (
    build_docstring,
    format_parameter,
    parse_arguments,
    parse_returns_annotation,
)
from .filter import FunctionFilter
from .type_override import TypeOverride


def generate_function(
    method: dict,
    is_static: bool = False,
    class_name: str | None = None,
) -> tuple[str | None, set[str]]:
    name = method["name"]

    if FunctionFilter.is_nuked(name, class_name):
        return None, set()

    desc = method.get("description", "").strip()
    args_block = method.get("arguments", [])
    returns_block = method.get("returns", [])

    args = parse_arguments(args_block)

    context = None
    if class_name:
        context = f"{class_name}:{name}"

    used_types = set()

    params = []
    for a in args:
        arg_type = a.get("type", "Any")
        arg_type = TypeOverride.get(arg_type, context)

        if arg_type not in ["vararg", "*args"]:
            used_types.add(arg_type)

        param_str = format_parameter(a, context)
        params.append(param_str)

    ret_ann = parse_returns_annotation(returns_block, context)
    if ret_ann != "None":
        used_types.add(ret_ann)

    lines = []
    if is_static:
        lines.append("    @staticmethod")
        lines.append(f"    def {name}({', '.join(params)}) -> {ret_ann}:")

    else:
        full_params = ["self"] + params
        lines.append(f"    def {name}({', '.join(full_params)}) -> {ret_ann}:")

    doc = build_docstring(desc, args, returns_block, base_indent=8, context=context)
    lines.append(doc)
    lines.append("        ...")

    return "\n".join(lines), used_types

from .arg_parser import (
    build_docstring,
    format_parameter,
    parse_arguments,
    parse_returns_annotation,
)


def generate_function(
    method: dict,
    is_static: bool = False,
    class_name: str | None = None,
) -> str:
    name = method["name"]
    desc = method.get("description", "").strip()
    args_block = method.get("arguments", [])
    returns_block = method.get("returns", [])

    args = parse_arguments(args_block)

    context = None
    if class_name:
        context = f"{class_name}:{name}"

    params = [format_parameter(a, context) for a in args]
    ret_ann = parse_returns_annotation(returns_block, context)

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
    return "\n".join(lines)

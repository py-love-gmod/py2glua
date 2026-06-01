from .utils import format_docstring


def generate(name: str, data: dict) -> str:
    lines = ["from __future__ import annotations", "", ""]
    args_block = data.get("arguments", [])
    returns_block = data.get("returns", [])

    params = []
    for group in args_block:
        for arg in group.get("args", []):
            if arg.get("name"):
                arg_type = arg.get("type", "Any")
                params.append(f"{arg['name']}: {arg_type}")

    if returns_block:
        ret_types = [ret.get("type", "Any") for ret in returns_block]
        ret_ann = (
            ret_types[0] if len(ret_types) == 1 else f"tuple[{', '.join(ret_types)}]"
        )

    else:
        ret_ann = "None"

    lines.append(f"def {name}({', '.join(params)}) -> {ret_ann}:")
    desc = data.get("description", "")
    if desc:
        lines.append(format_docstring(desc, 4))

    else:
        lines.append('    """..."""')

    lines.append("    ...")
    lines.append("")
    return "\n".join(lines)

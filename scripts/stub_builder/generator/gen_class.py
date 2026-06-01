from .utils import format_docstring


def generate(name: str, data: dict) -> str:
    lines = ["from __future__ import annotations", "", ""]
    class_name = name.capitalize() if name[0].islower() else name
    parent = data.get("parent")
    if parent:
        lines.append(f"class {class_name}({parent}):")

    else:
        lines.append(f"class {class_name}:")

    desc = data.get("description", "")

    if desc:
        lines.append(format_docstring(desc, 4))
    else:
        lines.append('    """..."""')

    lines.append("")

    for method in data.get("methods", []):
        mname = method["name"]
        mdesc = method.get("description", "").strip()
        args_block = method.get("arguments", [])
        returns_block = method.get("returns", [])

        params = ["self"]
        for group in args_block:
            for arg in group.get("args", []):
                if arg.get("name"):
                    arg_type = arg.get("type", "Any")
                    params.append(f"{arg['name']}: {arg_type}")

        if returns_block:
            ret_types = [ret.get("type", "Any") for ret in returns_block]
            ret_ann = (
                ret_types[0]
                if len(ret_types) == 1
                else f"tuple[{', '.join(ret_types)}]"
            )

        else:
            ret_ann = "None"

        lines.append(f"    def {mname}({', '.join(params)}) -> {ret_ann}:")
        if mdesc:
            lines.append(format_docstring(mdesc, 8))

        else:
            lines.append('        """..."""')

        lines.append("        ...")
        lines.append("")

    return "\n".join(lines)

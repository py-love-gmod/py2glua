from .utils import format_docstring


def parse_arguments(args_block: list) -> list:
    result = []
    for group in args_block:
        for arg in group.get("args", []):
            if arg.get("name"):
                result.append(arg)

    return result


def format_parameter(arg: dict) -> str:
    name = arg["name"]
    arg_type = arg.get("type", "Any")
    return f"{name}: {arg_type}"


def format_args_doc(args: list, base_indent: int = 8) -> str:
    if not args:
        return ""

    ind = " " * base_indent
    lines = [f"{ind}Args:"]
    for arg in args:
        name = arg["name"]
        arg_type = arg.get("type", "Any")
        desc = arg.get("description", "").strip()
        if desc:
            lines.append(f"{ind}    {name} ({arg_type}): {desc}")

        else:
            lines.append(f"{ind}    {name} ({arg_type})")

    return "\n".join(lines)


def format_returns_doc(returns_block: list, base_indent: int = 8) -> str:
    if not returns_block:
        return ""

    ind = " " * base_indent
    lines = [f"{ind}Returns:"]
    for ret in returns_block:
        ret_type = ret.get("type", "Any")
        desc = ret.get("description", "").strip()
        if desc:
            lines.append(f"{ind}    {ret_type}: {desc}")

        else:
            lines.append(f"{ind}    {ret_type}")

    return "\n".join(lines)


def parse_returns_annotation(returns_block: list) -> str:
    if not returns_block:
        return "None"

    types = [ret.get("type", "Any") for ret in returns_block]
    if len(types) == 1:
        return types[0]

    return f"tuple[{', '.join(types)}]"


def build_docstring(
    description: str,
    args: list,
    returns_block: list,
    base_indent: int = 8,
) -> str:
    if description:
        main_doc = format_docstring(description, base_indent)

    else:
        ind = " " * base_indent
        main_doc = f'{ind}"""..."""'

    if not args and not returns_block:
        return main_doc

    extra_parts = []
    args_doc = format_args_doc(args, base_indent)
    if args_doc:
        extra_parts.append(args_doc)

    ret_doc = format_returns_doc(returns_block, base_indent)
    if ret_doc:
        extra_parts.append(ret_doc)

    extra = "\n\n".join(extra_parts)

    lines = main_doc.splitlines()
    closing = lines.pop()
    lines.append("")
    lines.extend(extra.splitlines())
    lines.append(closing)  # pyright: ignore[reportArgumentType]

    return "\n".join(lines)

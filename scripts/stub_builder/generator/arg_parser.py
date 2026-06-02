from .type_override import TypeOverride
from .utils import format_docstring

PYTHON_KEYWORDS = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    "True",
    "False",
    "None",
    "type",
}


def escape_keyword(name: str) -> str:
    if name in PYTHON_KEYWORDS:
        return f"{name}_"

    return name


def parse_arguments(args_block: list) -> list:
    result = []
    for group in args_block:
        for arg in group.get("args", []):
            if arg.get("name"):
                arg["name"] = escape_keyword(arg["name"])
                result.append(arg)

    return result


def format_parameter(arg: dict, context: str | None = None) -> str:
    name = arg["name"].replace(" ", "_")
    arg_type = arg.get("type", "Any")

    arg_type = TypeOverride.get(arg_type, context)
    if arg_type == "vararg":
        return "*args"

    return f"{name}: {arg_type}"


def format_args_doc(
    args: list,
    base_indent: int = 8,
    context: str | None = None,
) -> str:
    if not args:
        return ""

    ind = " " * base_indent
    lines = [f"{ind}Args:"]
    for arg in args:
        name = arg["name"]
        arg_type = TypeOverride.get(arg.get("type", "Any"), context)

        if arg.get("type") == "vararg":
            display_name = "*args"
            display_type = "Any"

        else:
            display_name = name
            display_type = arg_type

        desc = arg.get("description", "").strip()
        if desc:
            lines.append(f"{ind}    {display_name} ({display_type}): {desc}")

        else:
            lines.append(f"{ind}    {display_name} ({display_type})")

    return "\n".join(lines)


def format_returns_doc(
    returns_block: list,
    base_indent: int = 8,
    context: str | None = None,
) -> str:
    if not returns_block:
        return ""

    ind = " " * base_indent
    lines = [f"{ind}Returns:"]
    for ret in returns_block:
        ret_type = TypeOverride.get(ret.get("type", "Any"), context)
        desc = ret.get("description", "").strip()
        if desc:
            lines.append(f"{ind}    {ret_type}: {desc}")

        else:
            lines.append(f"{ind}    {ret_type}")

    return "\n".join(lines)


def parse_returns_annotation(returns_block: list, context: str | None = None) -> str:
    if not returns_block:
        return "None"

    types = [TypeOverride.get(ret.get("type", "Any"), context) for ret in returns_block]
    if len(types) == 1:
        return types[0]

    return f"tuple[{', '.join(types)}]"


def build_docstring(
    description: str,
    args: list,
    returns_block: list,
    base_indent: int = 8,
    context: str | None = None,
) -> str:
    if description:
        main_doc = format_docstring(description, base_indent)

    else:
        ind = " " * base_indent
        main_doc = f'{ind}"""..."""'

    if not args and not returns_block:
        return main_doc

    extra_parts = []
    args_doc = format_args_doc(args, base_indent, context)
    if args_doc:
        extra_parts.append(args_doc)

    ret_doc = format_returns_doc(returns_block, base_indent, context)
    if ret_doc:
        extra_parts.append(ret_doc)

    extra = "\n\n".join(extra_parts)

    lines = main_doc.splitlines()
    closing = lines.pop()
    lines.append("")
    lines.extend(extra.splitlines())
    lines.append(closing)  # pyright: ignore[reportArgumentType]

    return "\n".join(lines)

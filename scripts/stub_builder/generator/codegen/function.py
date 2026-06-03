from ..function_filter import FunctionFilter
from ..models import FunctionDef
from ..type_override import TypeOverride
from .base import GeneratedCode
from .utils import format_docstring, sanitize_name


def build_args_doc(
    args: list,
    base_indent: int = 8,
    type_override: TypeOverride | None = None,
    context: str | None = None,
) -> str:
    if not args:
        return ""

    ind = " " * base_indent
    lines = [f"{ind}Args:"]
    for arg in args:
        name = sanitize_name(arg.name)
        arg_type = arg.type
        if type_override:
            arg_type = type_override.get(arg_type, context)

        if arg.kind == "vararg":
            display_name = f"*{name}"
            display_type = arg_type

        elif arg.kind == "kwarg":
            display_name = f"**{name}"
            display_type = arg_type

        else:
            display_name = name
            display_type = arg_type

        desc = " ".join(arg.description.split())
        if desc:
            lines.append(f"{ind}    {display_name} ({display_type}): {desc}")

        else:
            lines.append(f"{ind}    {display_name} ({display_type})")

    return "\n".join(lines)


def build_returns_doc(
    returns: list,
    base_indent: int = 8,
    type_override: TypeOverride | None = None,
    context: str | None = None,
) -> str:
    if not returns:
        return ""

    ind = " " * base_indent
    lines = [f"{ind}Returns:"]
    for ret in returns:
        ret_type = ret.type
        if type_override:
            ret_type = type_override.get(ret_type, context)

        desc = " ".join(ret.description.split())
        if desc:
            lines.append(f"{ind}    {ret_type}: {desc}")

        else:
            lines.append(f"{ind}    {ret_type}")

    return "\n".join(lines)


def build_docstring(
    description: str,
    args: list,
    returns: list,
    base_indent: int = 8,
    type_override: TypeOverride | None = None,
    context: str | None = None,
) -> str:
    if description:
        main_doc = format_docstring(description, base_indent)

    else:
        ind = " " * base_indent
        main_doc = f'{ind}"""\n{ind}    ...\n{ind}"""'

    if not args and not returns:
        return main_doc

    extra_parts = []
    args_doc = build_args_doc(args, base_indent, type_override, context)
    if args_doc:
        extra_parts.append(args_doc)

    ret_context = f"{context}.return" if context else None
    ret_doc = build_returns_doc(returns, base_indent, type_override, ret_context)
    if ret_doc:
        extra_parts.append(ret_doc)

    extra = "\n\n".join(extra_parts)

    lines = main_doc.splitlines()
    closing = lines.pop()
    lines.append("")
    lines.extend(extra.splitlines())
    lines.append(closing)  # pyright: ignore[reportArgumentType]
    return "\n".join(lines)


def parse_returns_annotation(
    returns: list,
    type_override: TypeOverride | None = None,
    context: str | None = None,
) -> str:
    if not returns:
        return "None"

    types = [
        type_override.get(r.type, context) if type_override else r.type for r in returns
    ]
    if len(types) == 1:
        return types[0]

    return f"tuple[{', '.join(types)}]"


def generate_function(
    method: FunctionDef,
    type_override: TypeOverride,
    function_filter: FunctionFilter,
    class_name: str | None = None,
) -> GeneratedCode | None:
    if function_filter.is_excluded(method.name, class_name):
        return None

    params: list[str] = []
    lines = []
    prefix = "    "
    used_types = set()
    valid_args = [arg for arg in method.arguments if arg.name.strip()]
    func_name = sanitize_name(method.name)
    context = f"{class_name}.{func_name}" if class_name else None

    if method.deprecated:
        if isinstance(method.deprecated, str):
            lines.append(
                f'{prefix}@deprecated("{method.deprecated.replace('"', r"\"")}")'
            )

        else:
            lines.append(f"{prefix}@deprecated")

        used_types.add("deprecated")

    for arg in valid_args:
        arg_type = type_override.get(arg.type, context)
        safe_name = sanitize_name(arg.name)

        if arg.kind == "vararg":
            params.append(f"*{safe_name}")

        elif arg.kind == "kwarg":
            params.append(f"**{safe_name}")

        else:
            if arg_type != "None":
                used_types.add(arg_type)

            params.append(f"{safe_name}: {arg_type}")

    ret_context = f"{context}.return" if context else None
    ret_ann = parse_returns_annotation(method.returns, type_override, ret_context)
    for r in method.returns:
        rt = type_override.get(r.type, ret_context)
        if rt != "None":
            used_types.add(rt)

    if method.is_static:
        lines.extend(
            [
                f"{prefix}@staticmethod",
                f"{prefix}def {func_name}({', '.join(params)}) -> {ret_ann}:",
            ]
        )

    else:
        if params:
            lines.append(
                f"{prefix}def {func_name}(self, {', '.join(params)}) -> {ret_ann}:"
            )

        else:
            lines.append(f"{prefix}def {func_name}(self) -> {ret_ann}:")

    doc = build_docstring(
        method.description,
        valid_args,
        method.returns,
        base_indent=8,
        type_override=type_override,
        context=context,
    )
    lines.append(doc)
    lines.append(f"{prefix}    ...")

    return GeneratedCode("\n".join(lines), used_types)

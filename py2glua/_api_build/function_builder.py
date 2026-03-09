from __future__ import annotations

import keyword
from typing import Any

from .._cli import CompilerExit
from .models import (
    BODY_DECORATOR,
    BODY_ELLIPSIS,
    BODY_PASS,
    BODY_RAW,
    ApiArgSpec,
    ApiDecoratorSpec,
    ApiFunctionSpec,
)


def render_functions(functions: list[ApiFunctionSpec], *, indent: int = 0) -> list[str]:
    out: list[str] = []
    for fn in functions:
        out.extend(render_function(fn, indent=indent))
        out.append("")
    while out and not out[-1].strip():
        out.pop()
    return out


def render_function(fn: ApiFunctionSpec, *, indent: int = 0) -> list[str]:
    pad = " " * indent
    out: list[str] = []

    if fn.deprecated:
        out.append(f"{pad}@deprecated({render_python_literal(fn.deprecated)})")

    for dec in fn.decorators:
        for line in render_decorator(dec, fn_name=fn.name):
            out.append(pad + line)

    sig = render_signature(fn.args, fn_name=fn.name)
    ret = f" -> {render_annotation(fn.returns)}" if fn.returns else ""
    out.append(f"{pad}def {fn.name}({sig}){ret}:")

    if fn.docstring:
        out.extend(render_docstring(fn.docstring, indent=indent + 4))

    for line in render_body(fn.body, body_lines=fn.body_lines):
        if line:
            out.append(pad + line)
        else:
            out.append("")

    return out


def render_signature(args: list[ApiArgSpec], *, fn_name: str) -> str:
    parts: list[str] = []
    seen_kwonly_marker = False
    seen_vararg = False

    for idx, arg in enumerate(args):
        if arg.kind not in {"posonly", "normal", "kwonly", "vararg", "kwarg"}:
            CompilerExit.user_error(
                f"arg kind ({fn_name}.{arg.name}) должен быть одним из: posonly, normal, kwonly, vararg, kwarg",
                show_path=False,
                show_pos=False,
            )

        if arg.kind == "kwonly" and not seen_kwonly_marker and not seen_vararg:
            parts.append("*")
            seen_kwonly_marker = True

        if arg.kind == "vararg":
            token = "*" + arg.name
            seen_vararg = True
            seen_kwonly_marker = True
        elif arg.kind == "kwarg":
            token = "**" + arg.name
        else:
            token = arg.name

        if arg.ann:
            token += f": {render_annotation(arg.ann)}"
        if arg.has_default:
            token += f" = {render_python_literal(arg.default)}"
        parts.append(token)

        next_kind = args[idx + 1].kind if idx + 1 < len(args) else None
        if arg.kind == "posonly" and next_kind != "posonly":
            parts.append("/")

    return ", ".join(parts)


def render_decorator(dec: ApiDecoratorSpec, *, fn_name: str) -> list[str]:
    kind = dec.kind

    if kind == "empty":
        return []

    if kind == "raw":
        expr = require_non_empty_str(dec.options.get("expr"), f"decorator raw expr ({fn_name})")
        if not expr.startswith("@"):
            expr = "@" + expr
        return [expr]

    if kind == "gmod_api":
        name = require_non_empty_str(dec.options.get("name"), f"gmod_api name ({fn_name})")
        realms = _parse_realms(dec.options.get("realms"), fn_name=fn_name)
        method = bool(dec.options.get("method", False))

        lines = [
            "@CompilerDirective.gmod_api(",
            f'    "{name}",',
            f"    realm=[{', '.join(f'Realm.{r}' for r in realms)}],",
            f"    method={str(method)},",
        ]

        base_args = dec.options.get("base_args")
        if base_args is not None:
            if not isinstance(base_args, list):
                CompilerExit.user_error(
                    f"gmod_api base_args ({fn_name}) должен быть списком",
                    show_path=False,
                    show_pos=False,
                )
            lines.append(f"    base_args=[{', '.join(render_python_literal(x) for x in base_args)}],")

        lines.append(")")
        return lines

    if kind == "register_arg":
        reg_type = require_non_empty_str(dec.options.get("reg_type"), f"register_arg reg_type ({fn_name})")
        index = dec.options.get("index", 0)
        if not isinstance(index, int):
            CompilerExit.user_error(
                f"register_arg index ({fn_name}) должен быть int",
                show_path=False,
                show_pos=False,
            )

        return [
            "@CompilerDirective.register_arg(",
            f"    index={index},",
            f'    reg_type="{reg_type}",',
            ")",
        ]

    CompilerExit.user_error(
        f"Неизвестный decorator kind '{kind}' (функция {fn_name}). "
        "Для нестандартного случая оставь kind=raw.",
        show_path=False,
        show_pos=False,
    )
    raise AssertionError("unreachable")


def render_body(kind: str, *, body_lines: list[str] | None = None) -> list[str]:
    if kind == BODY_PASS:
        return ["    pass"]
    if kind == BODY_ELLIPSIS:
        return ["    ..."]
    if kind == BODY_DECORATOR:
        return [
            "    def decorator(fn):",
            "        return fn",
            "",
            "    return decorator",
        ]
    if kind == BODY_RAW:
        if not body_lines:
            return ["    pass"]

        out: list[str] = []
        for line in body_lines:
            if line:
                out.append("    " + line)
            else:
                out.append("")
        return out

    CompilerExit.user_error(
        f"Неизвестный body kind: {kind}",
        show_path=False,
        show_pos=False,
    )
    raise AssertionError("unreachable")


def render_docstring(text: str, *, indent: int) -> list[str]:
    safe = text.replace('"""', '\\"\\"\\"')
    lines = safe.splitlines() or [safe]

    pad = " " * indent
    out = [f'{pad}"""']
    for line in lines:
        if line:
            out.append(f"{pad}{line}")
        else:
            out.append(pad)
    out.append(f'{pad}"""')
    return out


def render_annotation(ann: str) -> str:
    # With `from __future__ import annotations`, annotation expressions stay unevaluated.
    return ann


def render_python_literal(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _render_double_quoted_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(render_python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = ", ".join(
            f"{render_python_literal(k)}: {render_python_literal(v)}"
            for k, v in value.items()
        )
        return "{" + pairs + "}"
    return repr(value)


def _render_double_quoted_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def require_identifier(value: Any, field_name: str) -> str:
    s = require_non_empty_str(value, field_name)
    if not s.isidentifier() or keyword.iskeyword(s):
        CompilerExit.user_error(
            f"{field_name}: '{s}' не является корректным python identifier",
            show_path=False,
            show_pos=False,
        )
    return s


def require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        CompilerExit.user_error(
            f"{field_name}: должно быть строкой",
            show_path=False,
            show_pos=False,
        )

    out = value.strip()
    if not out:
        CompilerExit.user_error(
            f"{field_name}: пустое значение",
            show_path=False,
            show_pos=False,
        )
    return out


def require_rel_file_path(value: Any, field_name: str) -> str:
    raw = require_non_empty_str(value, field_name).replace("\\", "/")
    if raw.startswith("/") or raw.startswith("../") or "/../" in raw:
        CompilerExit.user_error(
            f"{field_name}: недопустимый путь '{value}'",
            show_path=False,
            show_pos=False,
        )
    return raw


def _parse_realms(raw: Any, *, fn_name: str) -> list[str]:
    if raw is None:
        CompilerExit.user_error(
            f"gmod_api realms ({fn_name}) не задан",
            show_path=False,
            show_pos=False,
        )

    values: list[str]
    if isinstance(raw, list):
        values = [require_non_empty_str(item, f"gmod_api realms ({fn_name})") for item in raw]
    else:
        values = [require_non_empty_str(raw, f"gmod_api realms ({fn_name})")]

    normalized: list[str] = []
    for value in values:
        cur = value.strip().upper()
        if cur not in {"SERVER", "CLIENT", "SHARED", "MENU"}:
            CompilerExit.user_error(
                f"gmod_api realms ({fn_name}): неизвестный realm '{value}'",
                show_path=False,
                show_pos=False,
            )
        normalized.append(cur)
    return normalized

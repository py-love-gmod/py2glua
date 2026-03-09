from __future__ import annotations

from .function_builder import (
    render_annotation,
    render_decorator,
    render_docstring,
    render_function,
    render_python_literal,
)
from .models import ApiClassIfSpec, ApiClassSpec, ApiFieldSpec


def render_classes(classes: list[ApiClassSpec], *, indent: int = 0) -> list[str]:
    out: list[str] = []
    for cls in classes:
        out.extend(render_class(cls, indent=indent))
        out.append("")
    while out and not out[-1].strip():
        out.pop()
    return out


def render_class(cls: ApiClassSpec, *, indent: int = 0) -> list[str]:
    out: list[str] = []
    pad = " " * indent

    if cls.deprecated:
        out.append(f"{pad}@deprecated({render_python_literal(cls.deprecated)})")

    for dec in cls.decorators:
        for line in render_decorator(dec, fn_name=cls.name):
            out.append(pad + line)

    if cls.bases:
        out.append(f"{pad}class {cls.name}({', '.join(cls.bases)}):")
    else:
        out.append(f"{pad}class {cls.name}:")

    body: list[str] = []

    if cls.docstring:
        body.extend(render_docstring(cls.docstring, indent=indent + 4))
        body.append("")

    for field in cls.fields:
        body.extend(render_field(field, indent=indent + 4))
        body.append("")

    for method in cls.methods:
        body.extend(render_function(method, indent=indent + 4))
        body.append("")

    for nested in cls.classes:
        body.extend(render_class(nested, indent=indent + 4))
        body.append("")

    for cond in cls.ifs:
        body.extend(render_class_if(cond, indent=indent + 4))
        body.append("")

    while body and not body[-1].strip():
        body.pop()

    if not body:
        out.append(f"{pad}    pass")
    else:
        out.extend(body)

    return out


def render_field(field: ApiFieldSpec, *, indent: int) -> list[str]:
    pad = " " * indent
    line = pad + field.name

    if field.ann:
        line += f": {render_annotation(field.ann)}"

    if field.has_value:
        line += f" = {render_python_literal(field.value)}"

    out = [line]
    if field.docstring:
        out.extend(render_docstring(field.docstring, indent=indent))
    return out


def render_class_if(block: ApiClassIfSpec, *, indent: int) -> list[str]:
    pad = " " * indent
    out: list[str] = [f"{pad}if {block.condition}:"]

    body: list[str] = []
    for field in block.fields:
        body.extend(render_field(field, indent=indent + 4))
        body.append("")

    for method in block.methods:
        body.extend(render_function(method, indent=indent + 4))
        body.append("")

    for nested in block.classes:
        body.extend(render_class(nested, indent=indent + 4))
        body.append("")

    while body and not body[-1].strip():
        body.pop()

    if not body:
        out.append(f"{pad}    pass")
    else:
        out.extend(body)

    return out

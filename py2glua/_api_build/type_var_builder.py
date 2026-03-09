from __future__ import annotations

from .models import ApiTypeVarSpec


def render_type_vars(type_vars: list[ApiTypeVarSpec], *, indent: int = 0) -> list[str]:
    pad = " " * indent
    out: list[str] = []
    for item in type_vars:
        out.append(f"{pad}{item.name} = {item.expression}")
    return out


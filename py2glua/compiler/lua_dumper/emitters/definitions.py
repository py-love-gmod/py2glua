from __future__ import annotations

from plg_reader import IRClassDef, IRFunctionDef

from .._core import _EMITTERS, _INDENT_STR
from .._core import emit as _emit
from ..lua_helpers import expr_to_str


def register():
    def emit_function_def(node: IRFunctionDef, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        for dec in node.decorators:
            lines.append(f"{prefix}-- @{expr_to_str(dec.expr)}")

        params = []
        for p in node.params:
            if p.kind == "star_arg":
                params.append(f"*{p.name}")

            elif p.kind == "kw_arg":
                params.append(f"**{p.name}")

            elif p.kind in ("slash", "star"):
                continue

            else:
                params.append(p.name)

        lines.append(f"{prefix}function {node.name}({', '.join(params)})")
        for stmt in node.body:
            _emit(stmt, lines, indent + 1)

        lines.append(f"{prefix}end")

    def emit_class_def(node: IRClassDef, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        for dec in node.decorators:
            lines.append(f"{prefix}-- @{expr_to_str(dec.expr)}")

        lines.append(f"{prefix}-- class {node.name}")
        if node.bases:
            bases_str = ", ".join(expr_to_str(b) for b in node.bases)
            lines.append(f"{prefix}-- extends {bases_str}")

        lines.append(f"{prefix}{node.name} = {{}}")
        for stmt in node.body:
            _emit(stmt, lines, indent)

        lines.append(f"{prefix}-- end class {node.name}")

    _EMITTERS[IRFunctionDef] = emit_function_def
    _EMITTERS[IRClassDef] = emit_class_def

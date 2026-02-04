from __future__ import annotations

from typing import Final

from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRClassDef,
    PyIRDecorator,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


class StripEnumsAndGmodSpecialEnumDefsPass:
    """
    After enum-folding:
      - removes whole enum class definitions:
          class X(Enum): ...
          class Y(IntEnum): ...
          class Z(StrEnum): ...
      - removes whole classes marked with:
          @CompilerDirective.gmod_special_enum(...)
    """

    _TARGET_DECORATORS: Final[set[str]] = {"gmod_special_enum"}
    _ENUM_BASE_NAMES: Final[set[str]] = {"Enum", "IntEnum", "StrEnum"}

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        compiler_directive_symbol_ids = (
            StripEnumsAndGmodSpecialEnumDefsPass._collect_symbol_ids(
                ctx, "CompilerDirective"
            )
        )

        enum_symbol_ids: set[int] = set()
        for n in StripEnumsAndGmodSpecialEnumDefsPass._ENUM_BASE_NAMES:
            enum_symbol_ids |= StripEnumsAndGmodSpecialEnumDefsPass._collect_symbol_ids(
                ctx, n
            )

        def is_compiler_directive_expr(expr: PyIRNode) -> bool:
            if isinstance(expr, PyIRVarUse):
                return expr.name == "CompilerDirective"

            if isinstance(expr, PyIRSymLink):
                try:
                    if expr.name == "CompilerDirective":
                        return True

                except Exception:
                    pass

                try:
                    return expr.symbol_id in compiler_directive_symbol_ids

                except Exception:
                    return False

            if isinstance(expr, PyIRAttribute):
                if expr.attr == "CompilerDirective":
                    return True

                return is_compiler_directive_expr(expr.value)

            if isinstance(expr, PyIREmitExpr):
                return (
                    expr.kind == PyIREmitKind.GLOBAL
                    and expr.name == "CompilerDirective"
                )

            return False

        def is_target_decorator(d: PyIRDecorator) -> bool:
            exp = d.exper
            if isinstance(exp, PyIRAttribute):
                if (
                    exp.attr
                    not in StripEnumsAndGmodSpecialEnumDefsPass._TARGET_DECORATORS
                ):
                    return False

                return is_compiler_directive_expr(exp.value)

            return False

        def is_python_enum_base_expr(base: PyIRNode) -> bool:
            if isinstance(base, PyIRVarUse):
                return (
                    base.name in StripEnumsAndGmodSpecialEnumDefsPass._ENUM_BASE_NAMES
                )

            if isinstance(base, PyIRSymLink):
                try:
                    if (
                        base.name
                        in StripEnumsAndGmodSpecialEnumDefsPass._ENUM_BASE_NAMES
                    ):
                        return True

                except Exception:
                    pass

                try:
                    return base.symbol_id in enum_symbol_ids

                except Exception:
                    return False

            if isinstance(base, PyIRAttribute):
                if (
                    base.attr
                    not in StripEnumsAndGmodSpecialEnumDefsPass._ENUM_BASE_NAMES
                ):
                    return False

                v = base.value

                if isinstance(v, PyIRVarUse):
                    return v.name == "enum"

                if isinstance(v, PyIRSymLink):
                    try:
                        return v.name == "enum"
                    except Exception:
                        return False

                return is_python_enum_base_expr(v)

            if isinstance(base, PyIREmitExpr):
                if base.kind in (PyIREmitKind.GLOBAL, PyIREmitKind.RAW):
                    return (
                        base.name
                        in StripEnumsAndGmodSpecialEnumDefsPass._ENUM_BASE_NAMES
                    )

                return False

            return False

        def should_drop_class(node: PyIRClassDef) -> bool:
            for d in node.decorators:
                if is_target_decorator(d):
                    return True

            for b in node.bases:
                if is_python_enum_base_expr(b):
                    return True

            return False

        def strip_stmt_list(stmts: list[PyIRNode]) -> list[PyIRNode]:
            out: list[PyIRNode] = []
            for s in stmts:
                out.extend(strip_stmt(s))

            return out

        def strip_stmt(node: PyIRNode) -> list[PyIRNode]:
            if isinstance(node, PyIRIf):
                node.body = strip_stmt_list(node.body)
                node.orelse = strip_stmt_list(node.orelse)
                return [node]

            if isinstance(node, PyIRWhile):
                node.body = strip_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRFor):
                node.body = strip_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRWith):
                node.body = strip_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRWithItem):
                return [node]

            if isinstance(node, PyIRFunctionDef):
                node.body = strip_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRClassDef):
                if should_drop_class(node):
                    return []

                node.body = strip_stmt_list(node.body)
                return [node]

            return [node]

        ir.body = strip_stmt_list(ir.body)
        return ir

    @staticmethod
    def _collect_symbol_ids(ctx: SymLinkContext, name: str) -> set[int]:
        ids: set[int] = set()

        try:
            ctx.ensure_module_index()

        except Exception:
            return ids

        try:
            module_names = set(ctx.module_name_by_path.values())

        except Exception:
            return ids

        for mn in module_names:
            try:
                sym = ctx.get_exported_symbol(mn, name)

            except Exception:
                sym = None

            if sym is None:
                continue

            try:
                ids.add(sym.id.value)

            except Exception:
                continue

        return ids

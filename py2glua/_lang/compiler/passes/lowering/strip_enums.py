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
from ..rewrite_utils import rewrite_stmt_block
from ..analysis.symlinks import PyIRSymLink, SymLinkContext
from ..common import (
    canon_path,
    collect_decl_symid_cache,
    collect_core_compiler_directive_local_symbol_ids,
    collect_core_compiler_directive_symbol_ids,
    collect_exported_symbol_ids,
    decorator_call_parts,
    is_expr_on_symbol_ids,
)


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
        compiler_directive_symbol_ids = set(
            collect_core_compiler_directive_symbol_ids(ctx)
        )
        if ir.path is not None:
            current_module = ctx.module_name_by_path.get(ir.path)
            if current_module:
                compiler_directive_symbol_ids |= (
                    collect_core_compiler_directive_local_symbol_ids(
                        ir,
                        ctx=ctx,
                        current_module=current_module,
                    )
                )

        enum_symbol_ids: set[int] = set()
        for n in StripEnumsAndGmodSpecialEnumDefsPass._ENUM_BASE_NAMES:
            enum_symbol_ids |= collect_exported_symbol_ids(ctx, n)
        gmod_special_enum_ids = {
            int(sid) for sid in getattr(ctx, "_gmod_special_enum_store", {}).keys()
        }
        fp = canon_path(ir.path)
        symid_by_decl = collect_decl_symid_cache(
            ctx,
            cache_field="_gmod_special_enum_symid_by_decl_cache",
        )

        def is_target_decorator(d: PyIRDecorator) -> bool:
            func_expr, _args_p, _args_kw = decorator_call_parts(d)
            if isinstance(func_expr, PyIRAttribute):
                if (
                    func_expr.attr
                    not in StripEnumsAndGmodSpecialEnumDefsPass._TARGET_DECORATORS
                ):
                    return False

                return is_expr_on_symbol_ids(
                    func_expr.value,
                    symbol_ids=compiler_directive_symbol_ids,
                    ctx=ctx,
                )

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
            if fp is not None and node.line is not None:
                sid = symid_by_decl.get((fp, node.line, node.name))
                if sid is not None and int(sid) in gmod_special_enum_ids:
                    return True

            for d in node.decorators:
                if is_target_decorator(d):
                    return True

            for b in node.bases:
                if is_python_enum_base_expr(b):
                    return True

            return False

        def strip_stmt_list(stmts: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(stmts, strip_stmt)

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

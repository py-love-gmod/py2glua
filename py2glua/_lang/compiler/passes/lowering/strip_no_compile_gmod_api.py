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


class StripNoCompileAndGmodApiDefsPass:
    """
    After FoldCompileTimeBoolConstsPass: remove entire def-nodes from output:

      - @CompilerDirective.no_compile()
      - @CompilerDirective.gmod_api(...)

    IMPORTANT:
      - Removes the whole PyIRFunctionDef / PyIRClassDef node, not just body.
      - Also removes methods (they are PyIRFunctionDef inside PyIRClassDef.body).
      - Works best after symlinks, but this pass is intended to run in lowering stage.
    """

    _TARGET_DECORATORS: Final[set[str]] = {"no_compile", "gmod_api"}

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        compiler_directive_symbol_ids = (
            StripNoCompileAndGmodApiDefsPass._collect_compiler_directive_symbol_ids(ctx)
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
                if exp.attr not in StripNoCompileAndGmodApiDefsPass._TARGET_DECORATORS:
                    return False

                return is_compiler_directive_expr(exp.value)

            return False

        def should_drop_def(node: PyIRNode) -> bool:
            if isinstance(node, (PyIRFunctionDef, PyIRClassDef)):
                for d in node.decorators:
                    if is_target_decorator(d):
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
                if should_drop_def(node):
                    return []

                node.body = strip_stmt_list(node.body)
                return [node]

            if isinstance(node, PyIRClassDef):
                if should_drop_def(node):
                    return []

                node.body = strip_stmt_list(node.body)
                return [node]

            return [node]

        ir.body = strip_stmt_list(ir.body)
        return ir

    @staticmethod
    def _collect_compiler_directive_symbol_ids(ctx: SymLinkContext) -> set[int]:
        """
        Best-effort extraction of exported symbol ids for 'CompilerDirective'
        across indexed modules. This is what makes aliases work (usually).

        If ctx doesn't support module indexing here, falls back to empty set.
        """
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
                sym = ctx.get_exported_symbol(mn, "CompilerDirective")

            except Exception:
                sym = None

            if sym is None:
                continue

            try:
                ids.add(sym.id.value)

            except Exception:
                continue

        return ids

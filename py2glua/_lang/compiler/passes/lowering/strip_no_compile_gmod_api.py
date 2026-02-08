from __future__ import annotations

from typing import Final

from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRClassDef,
    PyIRDecorator,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ..rewrite_utils import rewrite_stmt_block
from ..analysis.symlinks import SymLinkContext
from ..common import (
    collect_core_compiler_directive_local_symbol_ids,
    collect_core_compiler_directive_symbol_ids,
    decorator_call_parts,
    is_expr_on_symbol_ids,
)


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

        def is_target_decorator(d: PyIRDecorator) -> bool:
            func_expr, _args_p, _args_kw = decorator_call_parts(d)
            if isinstance(func_expr, PyIRAttribute):
                if (
                    func_expr.attr
                    not in StripNoCompileAndGmodApiDefsPass._TARGET_DECORATORS
                ):
                    return False

                return is_expr_on_symbol_ids(
                    func_expr.value,
                    symbol_ids=compiler_directive_symbol_ids,
                    ctx=ctx,
                )

            return False

        def should_drop_def(node: PyIRNode) -> bool:
            if isinstance(node, (PyIRFunctionDef, PyIRClassDef)):
                for d in node.decorators:
                    if is_target_decorator(d):
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

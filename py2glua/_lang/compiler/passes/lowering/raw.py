from __future__ import annotations

from typing import Final

from ....py.ir_dataclass import (
    PyIRCall,
    PyIRConstant,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRNode,
)
from ..rewrite_utils import (
    rewrite_expr_with_store_default,
    rewrite_stmt_block,
)
from ..analysis.symlinks import SymLinkContext
from ..common import (
    canon_path,
    collect_core_compiler_directive_local_symbol_ids,
    is_attr_on_symbol_ids,
)


class RewriteRawCallsPass:
    """
    Lowering-pass:
      Rewrites statement-only calls:

          CompilerDirective.raw("lua code")

      into:

          PyIREmitExpr(kind=RAW, name="lua code")

    Rules:
      - Срабатывает только когда база декоратора точно связана
        с `py2glua.glua.core.*.CompilerDirective`.
      - `raw()` is statement-only. If used in expression position -> error.
      - Requires exactly 1 positional arg: a string constant.
      - No kwargs.

    Why statement-only:
      raw is an "emit escape hatch" and should not participate in expression semantics.
    """

    _RAW_NAME: Final[str] = "raw"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("RewriteRawCallsPass.run ожидает PyIRFile")

        if ir.path is None:
            return ir

        ctx.ensure_module_index()
        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        if canon_path(ir.path) is None:
            return ir

        cd_local_symids = collect_core_compiler_directive_local_symbol_ids(
            ir=ir,
            ctx=ctx,
            current_module=current_module,
        )

        def is_raw_call_expr(n: PyIRNode) -> bool:
            if not isinstance(n, PyIRCall):
                return False
            return is_attr_on_symbol_ids(
                n.func,
                attr=cls._RAW_NAME,
                symbol_ids=cd_local_symids,
                ctx=ctx,
            )

        def extract_raw_code(call: PyIRCall) -> str:
            if call.args_kw:
                raise TypeError(
                    "CompilerDirective.raw не поддерживает keyword-аргументы"
                )

            if len(call.args_p) != 1:
                raise TypeError(
                    "CompilerDirective.raw ожидает ровно 1 позиционный аргумент (lua_code: str)"
                )

            arg = call.args_p[0]
            if not isinstance(arg, PyIRConstant) or not isinstance(arg.value, str):
                raise TypeError("CompilerDirective.raw ожидает строковый литерал")

            return arg.value

        def make_raw_emit(ref: PyIRNode, lua_code: str) -> PyIREmitExpr:
            return PyIREmitExpr(
                line=ref.line,
                offset=ref.offset,
                kind=PyIREmitKind.RAW,
                name=lua_code,
                args_p=[],
                args_kw={},
            )

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            if is_raw_call_expr(node):
                raise TypeError(
                    "CompilerDirective.raw() допускается только как отдельный statement"
                )

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            if isinstance(st, PyIRCall) and is_raw_call_expr(st):
                code = extract_raw_code(st)
                return [make_raw_emit(st, code)]

            return [rw_expr(st, False)]

        ir.body = rw_block(ir.body)
        return ir

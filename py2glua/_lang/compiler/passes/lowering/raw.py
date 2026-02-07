from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final, Iterator, Sequence

from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRDecorator,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRNode,
    PyIRReturn,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ...compiler_ir import PyIRFunctionExpr
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


class RewriteRawCallsPass:
    """
    Lowering-pass:
      Rewrites statement-only calls:

          CompilerDirective.raw("lua code")

      into:

          PyIREmitExpr(kind=RAW, name="lua code")

    Rules:
      - Matches ONLY when `CompilerDirective` is a proven local binding in THIS FILE:
          * local `class CompilerDirective:`
          * `from X import CompilerDirective`
          * `from X import CompilerDirective as CD`
      - `raw()` is statement-only. If used in expression position -> error.
      - Requires exactly 1 positional arg: a string constant.
      - No kwargs.

    Why statement-only:
      raw is an "emit escape hatch" and should not participate in expression semantics.
    """

    _CD_NAME: Final[str] = "CompilerDirective"
    _RAW_NAME: Final[str] = "raw"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("RewriteRawCallsPass.run expects PyIRFile")

        if ir.path is None:
            return ir

        ctx.ensure_module_index()
        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        file_path = cls._canon_path(ir.path)
        if file_path is None:
            return ir

        cd_local_symids = cls._collect_local_compiler_directive_symids(
            ir=ir,
            ctx=ctx,
            current_module=current_module,
            file_path=file_path,
        )
        if not cd_local_symids:
            return ir

        def is_raw_call_expr(n: PyIRNode) -> bool:
            if not isinstance(n, PyIRCall):
                return False

            f = n.func

            if not isinstance(f, PyIRAttribute):
                return False
            if f.attr != cls._RAW_NAME:
                return False

            base = f.value
            if not isinstance(base, PyIRSymLink):
                return False

            return int(base.symbol_id) in cd_local_symids

        def extract_raw_code(call: PyIRCall) -> str:
            if call.args_kw:
                raise TypeError(
                    "CompilerDirective.raw does not support keyword arguments"
                )

            if len(call.args_p) != 1:
                raise TypeError(
                    "CompilerDirective.raw expects exactly 1 positional argument (lua_code: str)"
                )

            arg = call.args_p[0]
            if not isinstance(arg, PyIRConstant) or not isinstance(arg.value, str):
                raise TypeError(
                    "CompilerDirective.raw expects a string literal constant"
                )

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

        def rw_expr(node: PyIRNode, *, store: bool) -> PyIRNode:

            if is_raw_call_expr(node):
                raise TypeError(
                    "CompilerDirective.raw() is statement-only and cannot be used as an expression"
                )

            if isinstance(node, PyIRAttribute):
                node.value = rw_expr(node.value, store=False)
                return node

            if isinstance(node, PyIRCall):
                node.func = rw_expr(node.func, store=False)
                node.args_p = [rw_expr(a, store=False) for a in node.args_p]
                node.args_kw = {
                    k: rw_expr(v, store=False) for k, v in node.args_kw.items()
                }
                return node

            if isinstance(node, PyIRAssign):
                node.targets = [rw_expr(t, store=True) for t in node.targets]
                node.value = rw_expr(node.value, store=False)
                return node

            if isinstance(node, PyIRAugAssign):
                node.target = rw_expr(node.target, store=True)
                node.value = rw_expr(node.value, store=False)
                return node

            if isinstance(node, PyIRWithItem):
                node.context_expr = rw_expr(node.context_expr, store=False)
                if node.optional_vars is not None:
                    node.optional_vars = rw_expr(node.optional_vars, store=True)
                return node

            if isinstance(node, PyIRReturn):
                if node.value is not None:
                    node.value = rw_expr(node.value, store=False)
                return node

            if isinstance(node, PyIRDecorator):
                node.exper = rw_expr(node.exper, store=False)
                node.args_p = [rw_expr(a, store=False) for a in node.args_p]
                node.args_kw = {
                    k: rw_expr(v, store=False) for k, v in node.args_kw.items()
                }
                return node

            if isinstance(node, PyIRFunctionExpr):
                node.body = rw_block(node.body)
                return node

            if is_dataclass(node):
                for f in fields(node):
                    v = getattr(node, f.name)
                    if isinstance(v, PyIRNode):
                        setattr(node, f.name, rw_expr(v, store=False))
                    elif isinstance(v, list):
                        setattr(
                            node,
                            f.name,
                            [
                                rw_expr(x, store=False)
                                if isinstance(x, PyIRNode)
                                else x
                                for x in v
                            ],
                        )
                    elif isinstance(v, dict):
                        setattr(
                            node,
                            f.name,
                            {
                                k: (
                                    rw_expr(x, store=False)
                                    if isinstance(x, PyIRNode)
                                    else x
                                )
                                for k, x in v.items()
                            },
                        )
                return node

            return node

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:

            if isinstance(st, PyIRCall) and is_raw_call_expr(st):
                code = extract_raw_code(st)
                return [make_raw_emit(st, code)]

            if isinstance(st, PyIRFunctionDef):
                st.decorators = [rw_expr(d, store=False) for d in st.decorators]  # type: ignore
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRClassDef):
                st.decorators = [rw_expr(d, store=False) for d in st.decorators]  # type: ignore
                st.bases = [rw_expr(b, store=False) for b in st.bases]
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRIf):
                st.test = rw_expr(st.test, store=False)
                st.body = rw_block(st.body)
                st.orelse = rw_block(st.orelse)
                return [st]

            if isinstance(st, PyIRWhile):
                st.test = rw_expr(st.test, store=False)
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRFor):
                st.target = rw_expr(st.target, store=True)
                st.iter = rw_expr(st.iter, store=False)
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRWith):
                for it in st.items:
                    rw_expr(it, store=False)
                st.body = rw_block(st.body)
                return [st]

            return [rw_expr(st, store=False)]

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            out: list[PyIRNode] = []
            for st in body:
                out.extend(rw_stmt(st))
            return out

        ir.body = rw_block(ir.body)
        return ir

    @staticmethod
    def _canon_path(p: Path | None) -> Path | None:
        if p is None:
            return None
        try:
            return p.resolve()
        except Exception:
            return p

    @staticmethod
    def _iter_imported_names(
        names: Sequence[str | tuple[str, str]] | None,
    ) -> Iterator[tuple[str, str]]:
        if not names:
            return
        for n in names:
            if isinstance(n, tuple):
                yield n[0], n[1]
            else:
                yield n, n

    @classmethod
    def _collect_local_compiler_directive_symids(
        cls,
        *,
        ir: PyIRFile,
        ctx: SymLinkContext,
        current_module: str,
        file_path: Path,
    ) -> set[int]:
        """
        Returns symids of names in this file that refer to CompilerDirective class:
          - local class def `class CompilerDirective:`
          - from-import `from X import CompilerDirective as CD`
          - from-import `from X import CompilerDirective`
        """
        out: set[int] = set()

        def local_symbol_id(name: str) -> int | None:
            sym = ctx.get_exported_symbol(current_module, name)
            return sym.id.value if sym is not None else None

        for n in ir.walk():
            if (
                isinstance(n, PyIRClassDef)
                and n.name == cls._CD_NAME
                and n.line is not None
            ):
                sid = local_symbol_id(cls._CD_NAME)
                if sid is not None:
                    out.add(int(sid))
                break

        for n in ir.walk():
            if not isinstance(n, PyIRImport):
                continue

            modules = getattr(n, "modules", None) or []
            mod = modules[0] if isinstance(modules, list) and modules else ""
            names = getattr(n, "names", None)

            if not n.if_from or not mod or not names:
                continue

            for orig, bound in cls._iter_imported_names(names):
                if orig != cls._CD_NAME:
                    continue
                sid = local_symbol_id(bound)
                if sid is not None:
                    out.add(int(sid))

        return out

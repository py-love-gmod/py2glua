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
    PyIRDecorator,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRNode,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ...compiler_ir import PyIRFunctionExpr
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


class CollectWithConditionClassesPass:
    """
    Lowering-pass (phase 1):
    Collects symids of classes decorated with @CompilerDirective.with_condition(),
    but ONLY when that decorator is proven to come from CompilerDirective binding
    in the current file (local class def or imported binding).

    Writes into ctx:
      - ctx._with_condition_class_symids: set[int]
      - ctx._with_condition_symid_by_decl_cache: dict[(Path, line, name) -> symid]
    """

    _SET_FIELD: Final[str] = "_with_condition_class_symids"
    _DECL_CACHE_FIELD: Final[str] = "_with_condition_symid_by_decl_cache"
    _WITH_COND_NAME: Final[str] = "with_condition"
    _CD_NAME: Final[str] = "CompilerDirective"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectWithConditionClassesPass.run expects PyIRFile")

        s = getattr(ctx, cls._SET_FIELD, None)
        if not isinstance(s, set):
            s = set()
            setattr(ctx, cls._SET_FIELD, s)

        if ir.path is None:
            return ir

        # Needed to resolve module name -> local exported symids.
        ctx.ensure_module_index()
        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        symid_by_decl = cls._get_symid_by_decl(ctx)
        fp = cls._canon_path(ir.path)
        if fp is None:
            return ir

        # Collect local symbol ids that are bound to CompilerDirective in THIS FILE
        cd_local_symids = cls._collect_local_compiler_directive_symids(
            ir=ir,
            ctx=ctx,
            current_module=current_module,
            file_path=fp,
            symid_by_decl=symid_by_decl,
        )
        if not cd_local_symids:
            # Without any known CompilerDirective binding, we refuse to match anything.
            return ir

        # Scan classes; collect those decorated with *CompilerDirective*.with_condition()
        for n in ir.walk():
            if not isinstance(n, PyIRClassDef):
                continue
            if n.line is None:
                continue

            if not cls._has_with_condition(n.decorators, cd_local_symids):
                continue

            class_symid = symid_by_decl.get((fp, n.line, n.name))
            if class_symid is None:
                continue

            s.add(int(class_symid))

            # Strip only the matched directive decorator (not any random ".with_condition")
            n.decorators = [
                d
                for d in n.decorators
                if not cls._is_with_condition_expr(d.exper, cd_local_symids)
            ]

        return ir

    @staticmethod
    def _canon_path(p: Path | None) -> Path | None:
        if p is None:
            return None
        try:
            return p.resolve()
        except Exception:
            return p

    @classmethod
    def _get_symid_by_decl(
        cls, ctx: SymLinkContext
    ) -> dict[tuple[Path, int, str], int]:
        cached = getattr(ctx, cls._DECL_CACHE_FIELD, None)
        if isinstance(cached, dict):
            return cached

        out: dict[tuple[Path, int, str], int] = {}
        for sym in ctx.symbols.values():
            if sym.decl_file is None or sym.decl_line is None:
                continue
            p = cls._canon_path(sym.decl_file)
            if p is None:
                continue
            out[(p, sym.decl_line, sym.name)] = sym.id.value

        setattr(ctx, cls._DECL_CACHE_FIELD, out)
        return out

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
        symid_by_decl: dict[tuple[Path, int, str], int],
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
                sid = symid_by_decl.get((file_path, n.line, n.name))
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

    @classmethod
    def _has_with_condition(
        cls,
        decorators: list[PyIRDecorator],
        cd_local_symids: set[int],
    ) -> bool:
        for d in decorators or []:
            if cls._is_with_condition_expr(d.exper, cd_local_symids):
                return True
        return False

    @classmethod
    def _is_with_condition_expr(
        cls,
        expr: PyIRNode,
        cd_local_symids: set[int],
    ) -> bool:
        # peel calls: @X() / @X(...)
        while isinstance(expr, PyIRCall):
            expr = expr.func

        # must be: <CompilerDirectiveBinding>.with_condition
        if not isinstance(expr, PyIRAttribute):
            return False
        if expr.attr != cls._WITH_COND_NAME:
            return False

        base = expr.value
        if isinstance(base, PyIRSymLink):
            return int(base.symbol_id) in cd_local_symids

        return False


class RewriteWithConditionBlocksPass:
    """
    Lowering-pass (phase 2):
    Rewrites:

        with Class.attr:
            body

    into:

        if Class.attr:
            body

    Condition source is any class symid collected by CollectWithConditionClassesPass.
    Supports multiple with-items by nesting (right-to-left):
        with A.x, B.y:
            body
      => if A.x:
            if B.y:
                body

    If with has mixed items (condition + other), it nests as:
        if Cond:
            with Other:
                body
    """

    _SET_FIELD: Final[str] = "_with_condition_class_symids"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("RewriteWithConditionBlocksPass.run expects PyIRFile")

        s = getattr(ctx, cls._SET_FIELD, None)
        if not isinstance(s, set) or not s:
            return ir

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            out: list[PyIRNode] = []
            for st in body:
                out.extend(rw_stmt(st))
            return out

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            if isinstance(st, PyIRFunctionDef):
                st.decorators = [rw_expr(d) for d in st.decorators]  # type: ignore
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRFunctionExpr):
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRClassDef):
                st.decorators = [rw_expr(d) for d in st.decorators]  # type: ignore
                st.bases = [rw_expr(b) for b in st.bases]
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRIf):
                st.test = rw_expr(st.test)
                st.body = rw_block(st.body)
                st.orelse = rw_block(st.orelse)
                return [st]

            if isinstance(st, PyIRWhile):
                st.test = rw_expr(st.test)
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRFor):
                st.target = rw_expr(st.target)
                st.iter = rw_expr(st.iter)
                st.body = rw_block(st.body)
                return [st]

            if isinstance(st, PyIRWith):
                inner = rw_block(st.body)

                rebuilt: list[PyIRNode] = inner
                for it in reversed(st.items):
                    rebuilt = cls._wrap_with_item(it, rebuilt, st, s)

                return rebuilt

            return [rw_generic_stmt(st)]

        def rw_generic_stmt(node: PyIRNode) -> PyIRNode:
            if isinstance(node, PyIRAssign):
                node.targets = [rw_expr(t) for t in node.targets]
                node.value = rw_expr(node.value)
                return node

            if isinstance(node, PyIRAugAssign):
                node.target = rw_expr(node.target)
                node.value = rw_expr(node.value)
                return node

            if is_dataclass(node):
                for f in fields(node):
                    v = getattr(node, f.name)
                    if isinstance(v, PyIRNode):
                        setattr(node, f.name, rw_expr(v))

                    elif isinstance(v, list):
                        setattr(
                            node,
                            f.name,
                            [rw_expr(x) if isinstance(x, PyIRNode) else x for x in v],
                        )

                    elif isinstance(v, dict):
                        setattr(
                            node,
                            f.name,
                            {
                                k: (rw_expr(x) if isinstance(x, PyIRNode) else x)
                                for k, x in v.items()
                            },
                        )

            return node

        def rw_expr(node: PyIRNode) -> PyIRNode:
            if isinstance(node, PyIRAttribute):
                node.value = rw_expr(node.value)
                return node

            if isinstance(node, PyIRCall):
                node.func = rw_expr(node.func)
                node.args_p = [rw_expr(a) for a in node.args_p]
                node.args_kw = {k: rw_expr(v) for k, v in node.args_kw.items()}
                return node

            if is_dataclass(node):
                for f in fields(node):
                    v = getattr(node, f.name)
                    if isinstance(v, PyIRNode):
                        setattr(node, f.name, rw_expr(v))

                    elif isinstance(v, list):
                        setattr(
                            node,
                            f.name,
                            [rw_expr(x) if isinstance(x, PyIRNode) else x for x in v],
                        )

                    elif isinstance(v, dict):
                        setattr(
                            node,
                            f.name,
                            {
                                k: (rw_expr(x) if isinstance(x, PyIRNode) else x)
                                for k, x in v.items()
                            },
                        )
            return node

        ir.body = rw_block(ir.body)
        return ir

    @staticmethod
    def _is_with_condition_item(
        it: PyIRWithItem,
        with_condition_class_symids: set[int],
    ) -> bool:
        ce = it.context_expr
        if not isinstance(ce, PyIRAttribute):
            return False

        if not isinstance(ce.value, PyIRSymLink):
            return False

        return int(ce.value.symbol_id) in with_condition_class_symids

    @classmethod
    def _wrap_with_item(
        cls,
        it: PyIRWithItem,
        inner: list[PyIRNode],
        src_with: PyIRWith,
        with_condition_class_symids: set[int],
    ) -> list[PyIRNode]:
        if it.optional_vars is not None:
            w = PyIRWith(
                line=src_with.line,
                offset=src_with.offset,
                items=[it],
                body=inner,
            )
            return [w]

        if cls._is_with_condition_item(it, with_condition_class_symids):
            test = it.context_expr
            out_if = PyIRIf(
                line=src_with.line,
                offset=src_with.offset,
                test=test,
                body=inner,
                orelse=[],
            )
            return [out_if]

        w = PyIRWith(
            line=src_with.line,
            offset=src_with.offset,
            items=[it],
            body=inner,
        )
        return [w]

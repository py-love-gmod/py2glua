from __future__ import annotations

from typing import Iterable

from ....compiler.passes.analysis.symlinks import PyIRSymLink, SymLinkContext
from ....py.ir_dataclass import (
    LuaNil,
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBinOP,
    PyIRBreak,
    PyIRCall,
    PyIRClassDef,
    PyIRComment,
    PyIRConstant,
    PyIRContinue,
    PyIRDecorator,
    PyIRDict,
    PyIRDictItem,
    PyIREmitExpr,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRList,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
    PyIRSet,
    PyIRSubscript,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarCreate,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)


class NilFoldPass:
    """
    Normalizes "nil" usage into a real IR constant.

    Goal:
      - replace reads of imported `nil` with `PyIRConstant(value=nil)`
      - replace `nil()` (no-arg call) with `PyIRConstant(value=nil)`
      - support `import py2glua.glua.core.types as t; t.nil` too

    This pass MUST run after symlink analysis/rewrite, because we rely on symbol_id.
    """

    TYPES_MODULE = "py2glua.glua.core.types"

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        if ir.path is None:
            return ir

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        def make_nil(ref: PyIRNode) -> PyIRConstant:
            return PyIRConstant(line=ref.line, offset=ref.offset, value=LuaNil)

        def iter_imported_names(
            names: list[str | tuple[str, str]],
        ) -> Iterable[tuple[str, str]]:
            for n in names:
                if isinstance(n, tuple):
                    yield n[0], n[1]

                else:
                    yield n, n

        def local_symbol_id(name: str) -> int | None:
            sym = ctx.get_exported_symbol(current_module, name)
            return sym.id.value if sym is not None else None

        nil_canon_sym = ctx.get_exported_symbol(NilFoldPass.TYPES_MODULE, "nil")
        nil_canon_id = nil_canon_sym.id.value if nil_canon_sym is not None else None

        nil_name_ids: set[int] = set()

        types_module_alias_ids: set[int] = set()

        for node in ir.walk():
            if not isinstance(node, PyIRImport):
                continue

            mod = ".".join(node.modules) if node.modules else ""
            if mod != NilFoldPass.TYPES_MODULE:
                continue

            if node.if_from:
                for orig, bound in iter_imported_names(node.names):
                    if orig != "nil":
                        continue

                    sid = local_symbol_id(bound)
                    if sid is not None:
                        nil_name_ids.add(sid)

                continue

            if node.names:
                for _orig, bound in iter_imported_names(node.names):
                    sid = local_symbol_id(bound)
                    if sid is not None:
                        types_module_alias_ids.add(sid)

            else:
                pass

        nil_ids: set[int] = set(nil_name_ids)
        if nil_canon_id is not None:
            nil_ids.add(nil_canon_id)

        if not nil_ids and not types_module_alias_ids:
            return ir

        def rw(node: PyIRNode, *, store: bool) -> PyIRNode:
            if isinstance(node, PyIRSymLink):
                if store or node.is_store:
                    return node

                if node.symbol_id in nil_ids:
                    return make_nil(node)

                return node

            if isinstance(node, PyIRCall):
                node.func = rw(node.func, store=False)
                node.args_p = [rw(a, store=False) for a in node.args_p]
                node.args_kw = {k: rw(v, store=False) for k, v in node.args_kw.items()}

                if not store and not node.args_p and not node.args_kw:
                    if (
                        isinstance(node.func, PyIRSymLink)
                        and node.func.symbol_id in nil_ids
                    ):
                        return make_nil(node)

                return node

            if isinstance(node, PyIRAttribute):
                node.value = rw(node.value, store=False)

                if store:
                    return node

                if (
                    node.attr == "nil"
                    and isinstance(node.value, PyIRSymLink)
                    and not node.value.is_store
                    and node.value.symbol_id in types_module_alias_ids
                ):
                    return make_nil(node)

                return node

            if isinstance(
                node,
                (
                    PyIRConstant,
                    PyIRVarUse,
                    PyIRVarCreate,
                    PyIRImport,
                    PyIRBreak,
                    PyIRContinue,
                    PyIRPass,
                    PyIRComment,
                ),
            ):
                return node

            if isinstance(node, PyIRUnaryOP):
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRBinOP):
                node.left = rw(node.left, store=False)
                node.right = rw(node.right, store=False)
                return node

            if isinstance(node, PyIRSubscript):
                node.value = rw(node.value, store=False)
                node.index = rw(node.index, store=False)
                return node

            if isinstance(node, PyIRList):
                node.elements = [rw(e, store=False) for e in node.elements]
                return node

            if isinstance(node, PyIRTuple):
                node.elements = [rw(e, store=False) for e in node.elements]
                return node

            if isinstance(node, PyIRSet):
                node.elements = [rw(e, store=False) for e in node.elements]
                return node

            if isinstance(node, PyIRDictItem):
                node.key = rw(node.key, store=False)
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRDict):
                node.items = [  # type: ignore[list-item]
                    rw(i, store=False) if isinstance(i, PyIRNode) else i
                    for i in node.items
                ]
                return node

            if isinstance(node, PyIRDecorator):
                node.exper = rw(node.exper, store=False)
                node.args_p = [rw(a, store=False) for a in node.args_p]
                node.args_kw = {k: rw(v, store=False) for k, v in node.args_kw.items()}
                return node

            if isinstance(node, PyIREmitExpr):
                node.args_p = [rw(a, store=False) for a in node.args_p]
                node.args_kw = {k: rw(v, store=False) for k, v in node.args_kw.items()}
                return node

            # ---- statements ----
            if isinstance(node, PyIRAssign):
                node.targets = [rw(t, store=True) for t in node.targets]
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRAugAssign):
                node.target = rw(node.target, store=True)
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRReturn):
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRIf):
                node.test = rw(node.test, store=False)
                node.body = [rw(s, store=False) for s in node.body]
                node.orelse = [rw(s, store=False) for s in node.orelse]
                return node

            if isinstance(node, PyIRWhile):
                node.test = rw(node.test, store=False)
                node.body = [rw(s, store=False) for s in node.body]
                return node

            if isinstance(node, PyIRFor):
                node.target = rw(node.target, store=True)
                node.iter = rw(node.iter, store=False)
                node.body = [rw(s, store=False) for s in node.body]
                return node

            if isinstance(node, PyIRWithItem):
                node.context_expr = rw(node.context_expr, store=False)
                if node.optional_vars is not None:
                    node.optional_vars = rw(node.optional_vars, store=True)
                return node

            if isinstance(node, PyIRWith):
                node.items = [  # type: ignore[list-item]
                    rw(i, store=False) if isinstance(i, PyIRNode) else i
                    for i in node.items
                ]
                node.body = [rw(s, store=False) for s in node.body]
                return node

            if isinstance(node, PyIRFunctionDef):
                node.decorators = [  # type: ignore[list-item]
                    rw(d, store=False) if isinstance(d, PyIRNode) else d
                    for d in node.decorators
                ]
                node.body = [rw(s, store=False) for s in node.body]
                return node

            if isinstance(node, PyIRClassDef):
                node.bases = [rw(b, store=False) for b in node.bases]
                node.decorators = [  # type: ignore[list-item]
                    rw(d, store=False) if isinstance(d, PyIRNode) else d
                    for d in node.decorators
                ]
                node.body = [rw(s, store=False) for s in node.body]
                return node

            return node

        ir.body = [rw(n, store=False) for n in ir.body]
        return ir

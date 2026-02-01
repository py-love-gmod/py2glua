from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Iterator, Sequence

from .....config import Py2GluaConfig
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRImport,
    PyIRNode,
    PyIRWithItem,
)
from ...compiler_ir import PyIRFunctionExpr
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


class FoldCompileTimeBoolConstsPass:
    CORE_MODULE = "py2glua.glua.core.compiler_directive"
    TYPING_MODULE = "typing"

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        if ir.path is None:
            return ir

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        def iter_imported_names(
            names: Sequence[str | tuple[str, str]] | None,
        ) -> Iterator[tuple[str, str]]:
            if not names:
                return
            for n in names:
                if isinstance(n, tuple):
                    yield n[0], n[1]
                else:
                    yield n, n

        def local_symbol_id(name: str) -> int | None:
            sym = ctx.get_exported_symbol(current_module, name)
            return sym.id.value if sym is not None else None

        def canonical_symbol_id(module_path: str, export_name: str) -> int | None:
            sym = ctx.get_exported_symbol(module_path, export_name)
            return sym.id.value if sym is not None else None

        def make_bool(value: bool, ref: PyIRNode) -> PyIRNode:
            return PyIRConstant(
                line=ref.line,
                offset=ref.offset,
                value=value,
            )

        compiler_directive_canon_id = canonical_symbol_id(
            FoldCompileTimeBoolConstsPass.CORE_MODULE, "CompilerDirective"
        )

        type_checking_local_ids: set[int] = set()
        typing_module_local_ids: set[int] = set()

        for n in ir.walk():
            if not isinstance(n, PyIRImport):
                continue

            modules = getattr(n, "modules", None) or []
            mod = modules[0] if isinstance(modules, list) and modules else ""
            names = getattr(n, "names", None)

            if (
                n.if_from
                and mod == FoldCompileTimeBoolConstsPass.TYPING_MODULE
                and names
            ):
                for orig, bound in iter_imported_names(names):
                    if orig == "TYPE_CHECKING":
                        sid = local_symbol_id(bound)
                        if sid is not None:
                            type_checking_local_ids.add(sid)

            if (not n.if_from) and mod == FoldCompileTimeBoolConstsPass.TYPING_MODULE:
                if names:
                    for _, bound in iter_imported_names(names):
                        sid = local_symbol_id(bound)
                        if sid is not None:
                            typing_module_local_ids.add(sid)

                else:
                    sid = local_symbol_id("typing")
                    if sid is not None:
                        typing_module_local_ids.add(sid)

        if (
            compiler_directive_canon_id is None
            and not type_checking_local_ids
            and not typing_module_local_ids
        ):
            return ir

        def rw(node: PyIRNode, *, store: bool) -> PyIRNode:
            if isinstance(node, PyIRSymLink):
                if store or node.is_store:
                    return node

                if node.symbol_id in type_checking_local_ids:
                    return make_bool(False, node)

                return node

            if isinstance(node, PyIRAttribute):
                node.value = rw(node.value, store=False)

                if store:
                    return node

                if isinstance(node.value, PyIRSymLink) and not node.value.is_store:
                    base_id = node.value.symbol_id

                    if (
                        compiler_directive_canon_id is not None
                        and node.attr == "DEBUG"
                        and base_id == compiler_directive_canon_id
                    ):
                        return make_bool(bool(Py2GluaConfig.debug), node)

                    if (
                        node.attr == "TYPE_CHECKING"
                        and base_id in typing_module_local_ids
                    ):
                        return make_bool(False, node)

                return node

            if isinstance(node, PyIRCall):
                node.func = rw(node.func, store=False)
                node.args_p = [rw(a, store=False) for a in node.args_p]
                node.args_kw = {k: rw(v, store=False) for k, v in node.args_kw.items()}
                return node

            if isinstance(node, PyIRAssign):
                node.targets = [rw(t, store=True) for t in node.targets]
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRAugAssign):
                node.target = rw(node.target, store=True)
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRFor):
                node.target = rw(node.target, store=True)
                node.iter = rw(node.iter, store=False)
                node.body = [rw(x, store=False) for x in node.body]
                return node

            if isinstance(node, PyIRWithItem):
                node.context_expr = rw(node.context_expr, store=False)
                if node.optional_vars is not None:
                    node.optional_vars = rw(node.optional_vars, store=True)
                return node

            if isinstance(node, PyIRFunctionDef):
                node.decorators = [rw(d, store=False) for d in node.decorators]  # pyright: ignore
                node.body = [rw(x, store=False) for x in node.body]
                return node

            if isinstance(node, PyIRFunctionExpr):
                node.body = [rw(x, store=False) for x in node.body]
                return node

            if isinstance(node, PyIRClassDef):
                node.decorators = [rw(d, store=False) for d in node.decorators]  # pyright: ignore
                node.bases = [rw(b, store=False) for b in node.bases]
                node.body = [rw(x, store=False) for x in node.body]
                return node

            if is_dataclass(node):
                for f in fields(node):
                    v = getattr(node, f.name)

                    if isinstance(v, PyIRNode):
                        setattr(node, f.name, rw(v, store=False))

                    elif isinstance(v, list):
                        setattr(
                            node,
                            f.name,
                            [
                                rw(x, store=False) if isinstance(x, PyIRNode) else x
                                for x in v
                            ],
                        )

                    elif isinstance(v, dict):
                        setattr(
                            node,
                            f.name,
                            {
                                k: rw(x, store=False) if isinstance(x, PyIRNode) else x
                                for k, x in v.items()
                            },
                        )
                return node

            return node

        ir.body = [rw(x, store=False) for x in ir.body]
        return ir

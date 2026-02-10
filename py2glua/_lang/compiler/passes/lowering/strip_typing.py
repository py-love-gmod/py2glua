from __future__ import annotations

from typing import Final

from ....compiler.compiler_ir import PyIRDo
from ....py.ir_dataclass import (
    PyIRAnnotatedAssign,
    PyIRAssign,
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRDecorator,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRNode,
    PyIRSubscript,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ..analysis.symlinks import PyIRSymLink, SymLinkContext
from ..common import (
    collect_local_imported_symbol_ids,
    collect_symbol_ids_in_modules,
    is_expr_on_symbol_ids,
    iter_imported_names,
)
from ..rewrite_utils import rewrite_expr_with_store_default, rewrite_stmt_block


class StripTypingRuntimeArtifactsPass:
    """
    Removes runtime typing artifacts from:
      - typing
      - typing_extensions
      - collections.abc

    What is stripped:
      - imports from these modules
      - class bases/decorators that reference these modules
      - assignments/calls whose RHS is typing-only expression

    Special rewrites:
      - cast(T, x) -> x
      - assert_type(x, T) -> x
      - reveal_type(x) -> x

    Important:
      this pass must run after FoldCompileTimeBoolConstsPass
      (so TYPE_CHECKING is already folded by dedicated pass).
    """

    _STRIPPED_MODULES: Final[tuple[str, ...]] = (
        "typing",
        "typing_extensions",
        "collections.abc",
    )
    _NOOP_CALL_VALUE_ARG_INDEX: Final[dict[str, int]] = {
        "cast": 1,
        "assert_type": 0,
        "reveal_type": 0,
    }

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        stripped_symbol_ids = cls._collect_stripped_symbol_ids(ctx)
        local_alias_to_module: dict[str, str] = {}

        current_module = None
        if ir.path is not None:
            current_module = ctx.module_name_by_path.get(ir.path)
            if current_module is not None:
                stripped_symbol_ids |= cls._collect_local_stripped_symbol_ids(
                    ir=ir,
                    ctx=ctx,
                    current_module=current_module,
                    alias_to_module=local_alias_to_module,
                )

        noop_call_symbol_ids = cls._collect_noop_call_symbol_ids(
            ctx=ctx,
            ir=ir,
            current_module=current_module,
        )

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(node, PyIRCall):
                node.func = rw_expr(node.func, False)
                node.args_p = [rw_expr(arg, False) for arg in node.args_p]
                node.args_kw = {k: rw_expr(v, False) for k, v in node.args_kw.items()}

                replacement = cls._rewrite_noop_typing_call(
                    node=node,
                    ctx=ctx,
                    noop_call_symbol_ids=noop_call_symbol_ids,
                    alias_to_module=local_alias_to_module,
                )
                if replacement is not None:
                    return replacement

                return node

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        def is_typing_ref(node: PyIRNode) -> bool:
            if stripped_symbol_ids and is_expr_on_symbol_ids(
                node,
                symbol_ids=stripped_symbol_ids,
                ctx=ctx,
            ):
                return True

            parts = cls._attr_parts(node)
            if not parts:
                return False

            root = parts[0]
            module_name = local_alias_to_module.get(root)
            if module_name is None:
                return False

            if module_name == "collections.abc" and root == "collections":
                return len(parts) >= 2 and parts[1] == "abc"

            return True

        def is_typing_only_expr(node: PyIRNode) -> bool:
            if is_typing_ref(node):
                return True

            if isinstance(node, PyIRAttribute):
                return is_typing_only_expr(node.value)

            if isinstance(node, PyIRSubscript):
                return is_typing_only_expr(node.value)

            if isinstance(node, PyIRCall):
                return is_typing_only_expr(node.func)

            return False

        def is_typing_decorator(dec: PyIRDecorator) -> bool:
            fn_expr = dec.exper
            if isinstance(fn_expr, PyIRCall):
                fn_expr = fn_expr.func
            return is_typing_only_expr(fn_expr)

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(node: PyIRNode) -> list[PyIRNode]:
            if isinstance(node, PyIRImport):
                return cls._rewrite_import(node)

            if isinstance(node, PyIRIf):
                node.test = rw_expr(node.test, False)
                node.body = rw_block(node.body)
                node.orelse = rw_block(node.orelse)
                return [node]

            if isinstance(node, PyIRWhile):
                node.test = rw_expr(node.test, False)
                node.body = rw_block(node.body)
                return [node]

            if isinstance(node, PyIRFor):
                node.target = rw_expr(node.target, True)
                node.iter = rw_expr(node.iter, False)
                node.body = rw_block(node.body)
                return [node]

            if isinstance(node, PyIRWith):
                items: list[PyIRWithItem] = []
                for item in node.items:
                    item.context_expr = rw_expr(item.context_expr, False)
                    if item.optional_vars is not None:
                        item.optional_vars = rw_expr(item.optional_vars, True)
                    items.append(item)
                node.items = items
                node.body = rw_block(node.body)
                return [node]

            if isinstance(node, PyIRDo):
                node.body = rw_block(node.body)
                return [node]

            if isinstance(node, PyIRFunctionDef):
                cleaned_decorators = []
                for dec in node.decorators:
                    dec.exper = rw_expr(dec.exper, False)
                    dec.args_p = [rw_expr(arg, False) for arg in dec.args_p]
                    dec.args_kw = {
                        key: rw_expr(value, False) for key, value in dec.args_kw.items()
                    }
                    if not is_typing_decorator(dec):
                        cleaned_decorators.append(dec)

                node.decorators = cleaned_decorators

                signature: dict[str, tuple[str | None, PyIRNode | None]] = {}
                for name, (ann, default) in node.signature.items():
                    if default is not None:
                        default = rw_expr(default, False)
                    signature[name] = (ann, default)
                node.signature = signature

                node.body = rw_block(node.body)
                return [node]

            if isinstance(node, PyIRClassDef):
                cleaned_decorators = []
                for dec in node.decorators:
                    dec.exper = rw_expr(dec.exper, False)
                    dec.args_p = [rw_expr(arg, False) for arg in dec.args_p]
                    dec.args_kw = {
                        key: rw_expr(value, False) for key, value in dec.args_kw.items()
                    }
                    if not is_typing_decorator(dec):
                        cleaned_decorators.append(dec)
                node.decorators = cleaned_decorators

                node.bases = [rw_expr(base, False) for base in node.bases]
                node.bases = [base for base in node.bases if not is_typing_only_expr(base)]
                node.body = rw_block(node.body)
                return [node]

            if isinstance(node, PyIRAssign):
                node.targets = [rw_expr(target, True) for target in node.targets]
                node.value = rw_expr(node.value, False)
                if is_typing_only_expr(node.value):
                    return []
                return [node]

            if isinstance(node, PyIRAnnotatedAssign):
                node.value = rw_expr(node.value, False)
                if is_typing_only_expr(node.value):
                    return []
                return [node]

            node = rw_expr(node, False)
            if is_typing_only_expr(node):
                return []
            return [node]

        ir.body = rw_block(ir.body)
        return ir

    @classmethod
    def _collect_stripped_symbol_ids(cls, ctx: SymLinkContext) -> set[int]:
        out: set[int] = set()
        for module_name in cls._STRIPPED_MODULES:
            exports = ctx.module_exports.get(module_name)
            if not exports:
                continue
            for symbol_id in exports.values():
                out.add(int(symbol_id.value))
        return out

    @classmethod
    def _collect_local_stripped_symbol_ids(
        cls,
        *,
        ir: PyIRFile,
        ctx: SymLinkContext,
        current_module: str,
        alias_to_module: dict[str, str],
    ) -> set[int]:
        out: set[int] = set()

        def local_symbol_id(name: str) -> int | None:
            symbol = ctx.get_exported_symbol(current_module, name)
            if symbol is None:
                return None
            return int(symbol.id.value)

        for node in ir.body:
            if not isinstance(node, PyIRImport):
                continue

            if node.if_from:
                module_name = ".".join(node.modules or ())
                for source_name, local_name in iter_imported_names(node.names):
                    full_module_name = (
                        f"{module_name}.{source_name}" if module_name else source_name
                    )
                    if (
                        module_name not in cls._STRIPPED_MODULES
                        and full_module_name not in cls._STRIPPED_MODULES
                    ):
                        continue

                    sid = local_symbol_id(local_name)
                    if sid is not None:
                        out.add(sid)
                    alias_to_module[local_name] = full_module_name
                continue

            if node.names:
                for source_name, local_name in iter_imported_names(node.names):
                    full_module_name = (
                        ".".join([*node.modules, source_name])
                        if node.modules
                        else source_name
                    )
                    if full_module_name not in cls._STRIPPED_MODULES:
                        continue

                    sid = local_symbol_id(local_name)
                    if sid is not None:
                        out.add(sid)
                    alias_to_module[local_name] = full_module_name
                continue

            full_module_name = ".".join(node.modules or ())
            if full_module_name not in cls._STRIPPED_MODULES or not node.modules:
                continue

            local_root = node.modules[0]
            sid = local_symbol_id(local_root)
            if sid is not None:
                out.add(sid)
            alias_to_module[local_root] = full_module_name

        return out

    @classmethod
    def _collect_noop_call_symbol_ids(
        cls,
        *,
        ctx: SymLinkContext,
        ir: PyIRFile,
        current_module: str | None,
    ) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for name in cls._NOOP_CALL_VALUE_ARG_INDEX:
            ids = set()
            ids |= collect_symbol_ids_in_modules(
                ctx,
                symbol_name=name,
                modules=cls._STRIPPED_MODULES,
            )
            if current_module is not None:
                ids |= collect_local_imported_symbol_ids(
                    ir=ir,
                    ctx=ctx,
                    current_module=current_module,
                    imported_name=name,
                    allowed_modules=cls._STRIPPED_MODULES,
                )
            out[name] = ids
        return out

    @classmethod
    def _rewrite_noop_typing_call(
        cls,
        *,
        node: PyIRCall,
        ctx: SymLinkContext,
        noop_call_symbol_ids: dict[str, set[int]],
        alias_to_module: dict[str, str],
    ) -> PyIRNode | None:
        for name, value_arg_index in cls._NOOP_CALL_VALUE_ARG_INDEX.items():
            if not cls._is_named_call(
                func=node.func,
                name=name,
                ctx=ctx,
                symbol_ids=noop_call_symbol_ids.get(name, set()),
                alias_to_module=alias_to_module,
            ):
                continue

            if len(node.args_p) <= value_arg_index:
                return None

            return node.args_p[value_arg_index]

        return None

    @classmethod
    def _is_named_call(
        cls,
        *,
        func: PyIRNode,
        name: str,
        ctx: SymLinkContext,
        symbol_ids: set[int],
        alias_to_module: dict[str, str],
    ) -> bool:
        if symbol_ids and is_expr_on_symbol_ids(func, symbol_ids=symbol_ids, ctx=ctx):
            return True

        if not isinstance(func, PyIRAttribute):
            return False

        if func.attr != name:
            return False

        parts = cls._attr_parts(func.value)
        if not parts:
            return False

        root = parts[0]
        module_name = alias_to_module.get(root)
        if module_name is None:
            return False

        if module_name == "collections.abc" and root == "collections":
            return len(parts) >= 2 and parts[1] == "abc"

        return True

    @classmethod
    def _rewrite_import(cls, node: PyIRImport) -> list[PyIRNode]:
        if node.if_from:
            module_name = ".".join(node.modules or ())
            kept_names: list[str | tuple[str, str]] = []
            for source_name, local_name in iter_imported_names(node.names):
                full_module_name = (
                    f"{module_name}.{source_name}" if module_name else source_name
                )
                if (
                    module_name in cls._STRIPPED_MODULES
                    or full_module_name in cls._STRIPPED_MODULES
                ):
                    continue
                kept_names.append(
                    source_name
                    if source_name == local_name
                    else (source_name, local_name)
                )

            if not kept_names:
                return []
            node.names = kept_names
            return [node]

        if node.names:
            kept_names = []
            for source_name, local_name in iter_imported_names(node.names):
                full_module_name = (
                    ".".join([*node.modules, source_name]) if node.modules else source_name
                )
                if full_module_name in cls._STRIPPED_MODULES:
                    continue
                kept_names.append(
                    source_name
                    if source_name == local_name
                    else (source_name, local_name)
                )

            if not kept_names:
                return []
            node.names = kept_names
            return [node]

        full_module_name = ".".join(node.modules or ())
        if full_module_name in cls._STRIPPED_MODULES:
            return []

        return [node]

    @classmethod
    def _attr_parts(cls, node: PyIRNode) -> list[str] | None:
        _ = cls
        attrs_rev: list[str] = []
        current = node
        while isinstance(current, PyIRAttribute):
            attrs_rev.append(current.attr)
            current = current.value

        if isinstance(current, PyIRSymLink):
            root_name = current.name
        elif isinstance(current, PyIRVarUse):
            root_name = current.name
        else:
            return None

        attrs_rev.reverse()
        return [root_name, *attrs_rev]


# Backward-compat alias for previous pass name.
StripTypingTypeVarProtocolPass = StripTypingRuntimeArtifactsPass

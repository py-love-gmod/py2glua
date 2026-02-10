from __future__ import annotations

from ....compiler.passes.analysis.symlinks import (
    PyIRSymLink,
    SymLinkContext,
    get_cached_symlink_usage_stats,
)
from ....py.ir_dataclass import (
    LuaNil,
    PyIRAnnotatedAssign,
    PyIRAnnotation,
    PyIRAssign,
    PyIRConstant,
    PyIRFile,
    PyIRImport,
    PyIRNode,
)
from ..common import canon_path, collect_decl_symid_cache, iter_imported_names
from ..rewrite_utils import rewrite_expr_with_store_default, rewrite_stmt_block


class InlineFinalLiteralPass:
    """
    Inlines reads of `Final` variables to literal constants when it is safe:

    - variable is annotated as `Final` (typing / typing_extensions, including aliases)
    - exactly one store in file IR
    - stored value is literal `PyIRConstant` (nil/None/bool/int/float/str)
    - variable is not used as call target (`x(...)` or nested call-func chain)
    """

    _DECL_CACHE_FIELD = "_final_inline_symid_by_decl_cache"
    _TYPING_MODULES: tuple[str, ...] = ("typing", "typing_extensions")

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        if ir.path is None:
            return ir

        file_path = canon_path(ir.path)
        if file_path is None:
            return ir

        final_ann_roots = cls._collect_final_annotation_roots(ir)
        decl_symid = collect_decl_symid_cache(ctx, cache_field=cls._DECL_CACHE_FIELD)

        final_symbol_ids: set[int] = set()
        decl_store_counts: dict[int, int] = {}
        literal_by_decl: dict[int, PyIRConstant] = {}

        for node in ir.walk():
            if isinstance(node, PyIRAnnotatedAssign):
                if not cls._is_final_annotation(node.annotation, final_ann_roots):
                    continue

                sid = cls._sid_by_decl(
                    decl_symid=decl_symid,
                    file_path=file_path,
                    line=node.line,
                    name=node.name,
                )
                if sid is None:
                    continue

                final_symbol_ids.add(sid)
                decl_store_counts[sid] = decl_store_counts.get(sid, 0) + 1

                lit = cls._literal_clone(node.value, ref=node)
                if lit is not None:
                    literal_by_decl[sid] = lit
                continue

            if isinstance(node, PyIRAnnotation):
                if not cls._is_final_annotation(node.annotation, final_ann_roots):
                    continue

                sid = cls._sid_by_decl(
                    decl_symid=decl_symid,
                    file_path=file_path,
                    line=node.line,
                    name=node.name,
                )
                if sid is not None:
                    final_symbol_ids.add(sid)

        if not final_symbol_ids:
            return ir

        usage = get_cached_symlink_usage_stats(ir, ctx)

        stores: dict[int, int] = {
            sid: max(
                int(usage.stores.get(sid, 0)),
                int(decl_store_counts.get(sid, 0)),
            )
            for sid in final_symbol_ids
        }
        reads: dict[int, int] = {
            sid: int(usage.reads.get(sid, 0)) for sid in final_symbol_ids
        }
        call_uses: dict[int, int] = {
            sid: int(usage.call_func_reads.get(sid, 0)) for sid in final_symbol_ids
        }

        literal_by_assign: dict[int, PyIRConstant] = {}
        non_literal_assign: set[int] = set()

        for node in ir.walk():
            if not isinstance(node, PyIRAssign):
                continue

            sid = cls._single_store_target_sid(node, final_symbol_ids)
            if sid is None:
                continue

            lit = cls._literal_clone(node.value, ref=node)
            if lit is None:
                non_literal_assign.add(sid)
            elif sid not in literal_by_assign:
                literal_by_assign[sid] = lit

        inline_values: dict[int, object | LuaNil] = {}
        for sid in sorted(final_symbol_ids):
            if stores.get(sid, 0) != 1:
                continue
            if call_uses.get(sid, 0) > 0:
                continue
            if sid in non_literal_assign:
                continue

            lit = literal_by_decl.get(sid) or literal_by_assign.get(sid)
            if lit is None:
                continue

            if reads.get(sid, 0) <= 0:
                continue

            inline_values[sid] = lit.value

        if not inline_values:
            return ir

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(node, PyIRSymLink):
                sid = int(node.symbol_id)
                if (store or node.is_store) or sid not in inline_values:
                    return node

                return PyIRConstant(
                    line=node.line,
                    offset=node.offset,
                    value=inline_values[sid],
                )

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

    @classmethod
    def _collect_final_annotation_roots(cls, ir: PyIRFile) -> set[str]:
        roots: set[str] = {"Final", "typing.Final", "typing_extensions.Final"}

        for node in ir.body:
            if not isinstance(node, PyIRImport):
                continue

            if node.if_from:
                module_name = ".".join(node.modules or ())
                if module_name not in cls._TYPING_MODULES:
                    continue

                for source_name, bound_name in iter_imported_names(node.names):
                    if source_name == "Final":
                        roots.add(bound_name)
                continue

            if node.names:
                for source_name, bound_name in iter_imported_names(node.names):
                    module_name = (
                        ".".join([*node.modules, source_name])
                        if node.modules
                        else source_name
                    )
                    if module_name in cls._TYPING_MODULES:
                        roots.add(f"{bound_name}.Final")
                continue

            module_name = ".".join(node.modules or ())
            if module_name in cls._TYPING_MODULES and node.modules:
                roots.add(f"{node.modules[0]}.Final")

        return roots

    @classmethod
    def _is_final_annotation(cls, annotation: str | None, roots: set[str]) -> bool:
        _ = cls
        if not annotation:
            return False

        text = annotation.replace(" ", "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()

        if not text:
            return False

        for root in roots:
            if text == root or text.startswith(f"{root}["):
                return True

        return False

    @staticmethod
    def _sid_by_decl(
        *,
        decl_symid: dict[tuple, int],
        file_path,
        line: int | None,
        name: str,
    ) -> int | None:
        if line is None:
            return None
        return decl_symid.get((file_path, line, name))

    @staticmethod
    def _is_foldable_constant_value(value: object | LuaNil) -> bool:
        if value is LuaNil or isinstance(value, LuaNil):
            return True
        if value is None:
            return True
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float, str)):
            return True
        return False

    @classmethod
    def _literal_clone(
        cls,
        node: PyIRNode,
        *,
        ref: PyIRNode,
    ) -> PyIRConstant | None:
        if not isinstance(node, PyIRConstant):
            return None
        if not cls._is_foldable_constant_value(node.value):
            return None
        return PyIRConstant(
            line=ref.line,
            offset=ref.offset,
            value=node.value,
        )

    @staticmethod
    def _single_store_target_sid(
        node: PyIRAssign,
        allowed: set[int],
    ) -> int | None:
        if len(node.targets) != 1:
            return None

        t = node.targets[0]
        if not isinstance(t, PyIRSymLink):
            return None

        sid = int(t.symbol_id)
        if sid not in allowed:
            return None

        if not t.is_store:
            return None

        return sid

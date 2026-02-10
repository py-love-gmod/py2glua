from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ....._config import Py2GluaConfig
from ....py.ir_dataclass import (
    LuaNil,
    PyIRAssign,
    PyIRAttribute,
    PyIRCall,
    PyIRComment,
    PyIRConstant,
    PyIRDecorator,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
)
from ...compiler_ir import PyIRDo, PyIRGoto, PyIRLabel
from ..common import CORE_COMPILER_DIRECTIVE_ATTR_PREFIXES
from .expand_context import ExpandContext, InlineTemplate

_KEY_SEP = "::"


def _attr_chain_parts(node: PyIRNode) -> List[str] | None:
    names_rev: List[str] = []
    cur: PyIRNode = node

    while isinstance(cur, PyIRAttribute):
        names_rev.append(cur.attr)
        cur = cur.value

    if not isinstance(cur, PyIRVarUse):
        return None

    names_rev.append(cur.name)
    names_rev.reverse()
    return names_rev


def _is_compiler_directive_attr(attr: PyIRAttribute, *, method: str) -> bool:
    if attr.attr != method:
        return False

    parts = _attr_chain_parts(attr.value)
    if parts is None:
        return False

    return tuple(parts) in CORE_COMPILER_DIRECTIVE_ATTR_PREFIXES


def _compiler_directive_kind(dec: PyIRDecorator) -> str | None:
    expr = dec.exper

    if isinstance(expr, PyIRAttribute):
        if _is_compiler_directive_attr(expr, method="inline"):
            return "inline"
        return None

    if isinstance(expr, PyIRCall):
        f = expr.func
        if isinstance(f, PyIRAttribute):
            if _is_compiler_directive_attr(f, method="inline"):
                return "inline"
        return None

    return None


def _is_inline_def(fn: PyIRFunctionDef) -> bool:
    return any(_compiler_directive_kind(d) == "inline" for d in fn.decorators)


def _collect_target_names(node: PyIRNode, out: Set[str]) -> None:
    if isinstance(node, PyIRVarUse):
        out.add(node.name)


def _collect_locals_in_body(stmts: List[PyIRNode], out: Set[str]) -> None:
    for st in stmts:
        if isinstance(st, PyIRAssign):
            for t in st.targets:
                _collect_target_names(t, out)
            continue

        if isinstance(st, PyIRFor):
            _collect_target_names(st.target, out)
            _collect_locals_in_body(st.body, out)
            continue

        if isinstance(st, PyIRIf):
            _collect_locals_in_body(st.body, out)
            _collect_locals_in_body(st.orelse, out)
            continue

        if isinstance(st, PyIRWhile):
            _collect_locals_in_body(st.body, out)
            continue

        if isinstance(st, PyIRWith):
            _collect_locals_in_body(st.body, out)
            continue

        if isinstance(st, PyIRFunctionDef):
            continue


def _module_parts_from_path(path: Path) -> tuple[str, ...] | None:
    p = path.resolve()

    parts = p.parts
    try:
        src_root = Py2GluaConfig.source.resolve()
        rel = p.relative_to(src_root)
    except Exception:
        rel = None

    if rel is None:
        try:
            i = parts.index("py2glua")
            tail = list(parts[i:])
        except ValueError:
            return None
    else:
        tail = list(rel.parts)

    if not tail:
        return None

    last = tail[-1]
    if last.endswith(".py"):
        name = last[:-3]
        tail[-1] = "__init__" if name == "__init__" else name

    return tuple(tail)


def _normalize_mod_parts(mod_parts: tuple[str, ...]) -> tuple[str, ...]:
    # NormalizeImportsPass works with package path without trailing __init__
    if mod_parts and mod_parts[-1] == "__init__":
        return mod_parts[:-1]
    return mod_parts


def _ir_mod_parts(ir: PyIRFile) -> tuple[str, ...] | None:
    if ir.path is None:
        return None
    mp = _module_parts_from_path(ir.path)
    if mp is None:
        return None
    return _normalize_mod_parts(mp)


def _qual_key(mod_parts: tuple[str, ...], fn_name: str) -> str:
    return f"{'.'.join(mod_parts)}{_KEY_SEP}{fn_name}"


def _call_inline_key(
    ir_mod_parts: tuple[str, ...] | None, func: PyIRNode
) -> str | None:
    """
    Deterministic keying:
      - local call:    from_meters(10)   -> "<this_module>::from_meters"
      - imported call: pkg.mod.fn(10)    -> "pkg.mod::fn"
    """
    if isinstance(func, PyIRVarUse):
        if ir_mod_parts is None:
            return None
        return _qual_key(ir_mod_parts, func.name)

    if isinstance(func, PyIRAttribute):
        parts = _attr_chain_parts(func)
        if parts is None or len(parts) < 2:
            return None
        mod = _normalize_mod_parts(tuple(parts[:-1]))
        name = parts[-1]
        if not mod:
            return None
        return _qual_key(mod, name)

    return None


class CollectInlineFunctionsPass:
    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> None:
        mod_parts = _ir_mod_parts(ir)

        for st in ir.body:
            if not isinstance(st, PyIRFunctionDef):
                continue

            if not _is_inline_def(st):
                continue

            params_list = list(st.signature.keys())
            if st.vararg is not None:
                params_list.append(st.vararg)
            if st.kwarg is not None:
                params_list.append(st.kwarg)
            params = tuple(params_list)

            locals_set: Set[str] = set(params)
            _collect_locals_in_body(st.body, locals_set)

            # Strong key: module-qualified, no collisions between modules.
            if mod_parts is not None:
                key = _qual_key(mod_parts, st.name)
                ctx.inline_templates[key] = InlineTemplate(
                    fn=st,
                    params=params,
                    locals=tuple(sorted(locals_set)),
                )
            else:
                # If module path is unknown, we can't safely qualify.
                continue


def _mk_var(name: str, ref: PyIRNode) -> PyIRVarUse:
    return PyIRVarUse(line=ref.line, offset=ref.offset, name=name)


def _mk_assign(target: PyIRNode, value: PyIRNode, ref: PyIRNode) -> PyIRAssign:
    return PyIRAssign(
        line=ref.line,
        offset=ref.offset,
        targets=[target],
        value=value,
    )


def _mk_nil(ref: PyIRNode) -> PyIRConstant:
    return PyIRConstant(line=ref.line, offset=ref.offset, value=LuaNil)


def _mk_do(body: List[PyIRNode], ref: PyIRNode) -> PyIRDo:
    return PyIRDo(line=ref.line, offset=ref.offset, body=body)


def _mk_label(name: str, ref: PyIRNode) -> PyIRLabel:
    return PyIRLabel(line=ref.line, offset=ref.offset, name=name)


def _mk_goto(label: str, ref: PyIRNode) -> PyIRGoto:
    return PyIRGoto(line=ref.line, offset=ref.offset, label=label)


def _rename_vars(node: PyIRNode, rename: Dict[str, str]) -> PyIRNode:
    if isinstance(node, PyIRVarUse):
        nn = rename.get(node.name)
        if nn is None:
            return deepcopy(node)
        return PyIRVarUse(line=node.line, offset=node.offset, name=nn)

    if not is_dataclass(node):
        return deepcopy(node)

    clone = deepcopy(node)
    for f in fields(clone):
        v = getattr(clone, f.name)

        if isinstance(v, PyIRNode):
            setattr(clone, f.name, _rename_vars(v, rename))
            continue

        if isinstance(v, list):
            new_list: list[object] = []
            for item in v:
                if isinstance(item, PyIRNode):
                    new_list.append(_rename_vars(item, rename))
                else:
                    new_list.append(item)
            setattr(clone, f.name, new_list)
            continue

        if isinstance(v, dict):
            new_dict: dict[object, object] = {}
            for key, item in v.items():
                new_key = (
                    _rename_vars(key, rename) if isinstance(key, PyIRNode) else key
                )
                new_item = (
                    _rename_vars(item, rename) if isinstance(item, PyIRNode) else item
                )
                new_dict[new_key] = new_item
            setattr(clone, f.name, new_dict)

    return clone


def _has_any_return(stmts: List[PyIRNode]) -> bool:
    for st in stmts:
        if isinstance(st, PyIRReturn):
            return True

        if isinstance(st, PyIRIf):
            if _has_any_return(st.body) or _has_any_return(st.orelse):
                return True

        elif isinstance(st, PyIRWhile):
            if _has_any_return(st.body):
                return True

        elif isinstance(st, PyIRFor):
            if _has_any_return(st.body):
                return True

        elif isinstance(st, PyIRWith):
            if _has_any_return(st.body):
                return True

    return False


def _has_return_inside_nested(stmts: List[PyIRNode]) -> bool:
    for st in stmts:
        if isinstance(st, PyIRReturn):
            return True

        if isinstance(st, PyIRIf):
            if _has_return_inside_nested(st.body) or _has_return_inside_nested(
                st.orelse
            ):
                return True

        elif isinstance(st, PyIRWhile):
            if _has_return_inside_nested(st.body):
                return True

        elif isinstance(st, PyIRFor):
            if _has_return_inside_nested(st.body):
                return True

        elif isinstance(st, PyIRWith):
            if _has_return_inside_nested(st.body):
                return True

    return False


def _needs_label(stmts: List[PyIRNode]) -> bool:
    for i, st in enumerate(stmts):
        if isinstance(st, PyIRReturn):
            if i != len(stmts) - 1:
                return True

        if isinstance(st, PyIRIf):
            if _has_return_inside_nested(st.body) or _has_return_inside_nested(
                st.orelse
            ):
                return True

        elif isinstance(st, PyIRWhile):
            if _has_return_inside_nested(st.body):
                return True

        elif isinstance(st, PyIRFor):
            if _has_return_inside_nested(st.body):
                return True

        elif isinstance(st, PyIRWith):
            if _has_return_inside_nested(st.body):
                return True

    return False


def _rewrite_returns_no_label(
    stmts: List[PyIRNode],
    *,
    ret_store: PyIRNode | None,
    ref: PyIRNode,
) -> List[PyIRNode]:
    out: List[PyIRNode] = []
    for st in stmts:
        if isinstance(st, PyIRReturn):
            if ret_store is not None:
                ret_value = deepcopy(st.value) if st.value is not None else _mk_nil(ref)
                out.append(_mk_assign(ret_store, ret_value, ref))
            continue

        out.append(st)

    return out


def _rewrite_returns_with_label(
    stmts: List[PyIRNode],
    *,
    ret_store: PyIRNode | None,
    end_label: str,
    ref: PyIRNode,
    top_level: bool,
) -> List[PyIRNode]:
    out: List[PyIRNode] = []

    for st in stmts:
        if isinstance(st, PyIRReturn):
            if ret_store is not None:
                ret_value = deepcopy(st.value) if st.value is not None else _mk_nil(ref)
                out.append(_mk_assign(ret_store, ret_value, ref))
            out.append(_mk_goto(end_label, ref))
            continue

        if isinstance(st, PyIRIf):
            st.body = _rewrite_returns_with_label(
                st.body,
                ret_store=ret_store,
                end_label=end_label,
                ref=ref,
                top_level=False,
            )
            st.orelse = _rewrite_returns_with_label(
                st.orelse,
                ret_store=ret_store,
                end_label=end_label,
                ref=ref,
                top_level=False,
            )
            out.append(st)
            continue

        if isinstance(st, PyIRWhile):
            st.body = _rewrite_returns_with_label(
                st.body,
                ret_store=ret_store,
                end_label=end_label,
                ref=ref,
                top_level=False,
            )
            out.append(st)
            continue

        if isinstance(st, PyIRFor):
            st.body = _rewrite_returns_with_label(
                st.body,
                ret_store=ret_store,
                end_label=end_label,
                ref=ref,
                top_level=False,
            )
            out.append(st)
            continue

        if isinstance(st, PyIRWith):
            st.body = _rewrite_returns_with_label(
                st.body,
                ret_store=ret_store,
                end_label=end_label,
                ref=ref,
                top_level=False,
            )
            out.append(st)
            continue

        out.append(st)

    if (
        top_level
        and out
        and isinstance(out[-1], PyIRGoto)
        and out[-1].label == end_label
    ):
        out.pop()

    return out


def _strip_nops(stmts: List[PyIRNode]) -> List[PyIRNode]:
    out: List[PyIRNode] = []
    for s in stmts:
        if isinstance(s, (PyIRComment, PyIRPass)):
            continue
        out.append(s)
    return out


def _count_var_uses(node: PyIRNode, out: Dict[str, int]) -> None:
    if isinstance(node, PyIRVarUse):
        out[node.name] = out.get(node.name, 0) + 1
        return

    if not is_dataclass(node):
        return

    for f in fields(node):
        v = getattr(node, f.name)

        if isinstance(v, PyIRNode):
            _count_var_uses(v, out)
            continue

        if isinstance(v, list):
            for x in v:
                if isinstance(x, PyIRNode):
                    _count_var_uses(x, out)
            continue

        if isinstance(v, dict):
            for x in v.values():
                if isinstance(x, PyIRNode):
                    _count_var_uses(x, out)
            continue


def _has_call(node: PyIRNode) -> bool:
    if isinstance(node, PyIRCall):
        return True

    if not is_dataclass(node):
        return False

    for f in fields(node):
        v = getattr(node, f.name)

        if isinstance(v, PyIRNode):
            if _has_call(v):
                return True
            continue

        if isinstance(v, list):
            for x in v:
                if isinstance(x, PyIRNode) and _has_call(x):
                    return True
            continue

        if isinstance(v, dict):
            for x in v.values():
                if isinstance(x, PyIRNode) and _has_call(x):
                    return True
            continue

    return False


def _subst_vars(node: PyIRNode, subst: Dict[str, PyIRNode]) -> PyIRNode:
    """
    Replace VarUse(param) -> deepcopy(arg_expr).
    Other nodes are deep-copied structurally.
    """
    if isinstance(node, PyIRVarUse):
        repl = subst.get(node.name)
        if repl is None:
            return deepcopy(node)
        return deepcopy(repl)

    if not is_dataclass(node):
        return deepcopy(node)

    kw: Dict[str, object] = {}
    for f in fields(node):
        v = getattr(node, f.name)

        if isinstance(v, PyIRNode):
            kw[f.name] = _subst_vars(v, subst)
            continue

        if isinstance(v, list):
            new_list: List[object] = []
            for x in v:
                if isinstance(x, PyIRNode):
                    new_list.append(_subst_vars(x, subst))
                else:
                    new_list.append(deepcopy(x))
            kw[f.name] = new_list
            continue

        if isinstance(v, dict):
            new_dict: Dict[object, object] = {}
            for k, x in v.items():
                if isinstance(x, PyIRNode):
                    new_dict[k] = _subst_vars(x, subst)
                else:
                    new_dict[k] = deepcopy(x)
            kw[f.name] = new_dict
            continue

        kw[f.name] = deepcopy(v)

    out = deepcopy(node)
    for field_name, field_value in kw.items():
        setattr(out, field_name, field_value)
    return out


def _collect_remaining_inline_call_keys(
    node: PyIRNode,
    *,
    ir_mod_parts: tuple[str, ...] | None,
    out: Set[str],
) -> None:
    if isinstance(node, PyIRCall):
        k = _call_inline_key(ir_mod_parts, node.func)
        if k is not None:
            out.add(k)

    if not is_dataclass(node):
        return

    for f in fields(node):
        v = getattr(node, f.name)

        if isinstance(v, PyIRNode):
            _collect_remaining_inline_call_keys(v, ir_mod_parts=ir_mod_parts, out=out)
            continue

        if isinstance(v, list):
            for x in v:
                if isinstance(x, PyIRNode):
                    _collect_remaining_inline_call_keys(
                        x, ir_mod_parts=ir_mod_parts, out=out
                    )
            continue

        if isinstance(v, dict):
            for x in v.values():
                if isinstance(x, PyIRNode):
                    _collect_remaining_inline_call_keys(
                        x, ir_mod_parts=ir_mod_parts, out=out
                    )
            continue


class RewriteInlineCallsPass:
    @staticmethod
    def run(ir: PyIRFile, ctx: ExpandContext) -> PyIRFile:
        mod_parts = _ir_mod_parts(ir)
        ir.body = RewriteInlineCallsPass._rewrite_stmt_list(ir, ir.body, ctx, mod_parts)

        # Strip inline defs only if there are no remaining call sites.
        RewriteInlineCallsPass._strip_fully_inlined_defs(ir, ctx, mod_parts)

        return ir

    @staticmethod
    def _strip_fully_inlined_defs(
        ir: PyIRFile,
        ctx: ExpandContext,
        mod_parts: tuple[str, ...] | None,
    ) -> None:
        if mod_parts is None:
            return

        remaining_calls: Set[str] = set()
        for st in ir.body:
            _collect_remaining_inline_call_keys(
                st, ir_mod_parts=mod_parts, out=remaining_calls
            )

        new_body: List[PyIRNode] = []
        for st in ir.body:
            if isinstance(st, PyIRFunctionDef) and _is_inline_def(st):
                key = _qual_key(mod_parts, st.name)
                # Strip only when:
                #  - this fn is actually registered as inline template
                #  - and there are no remaining calls to it after rewrite
                if key in ctx.inline_templates and key not in remaining_calls:
                    continue
            new_body.append(st)

        ir.body = new_body

    @staticmethod
    def _rewrite_stmt_list(
        ir: PyIRFile,
        stmts: List[PyIRNode],
        ctx: ExpandContext,
        mod_parts: tuple[str, ...] | None,
    ) -> List[PyIRNode]:
        out: List[PyIRNode] = []
        for st in stmts:
            out.extend(RewriteInlineCallsPass._rewrite_stmt(ir, st, ctx, mod_parts))
        return out

    @staticmethod
    def _rewrite_stmt(
        ir: PyIRFile,
        st: PyIRNode,
        ctx: ExpandContext,
        mod_parts: tuple[str, ...] | None,
    ) -> List[PyIRNode]:
        if isinstance(st, PyIRIf):
            pro, test = RewriteInlineCallsPass._rewrite_expr(
                ir, st.test, ctx, mod_parts
            )
            st.test = test
            st.body = RewriteInlineCallsPass._rewrite_stmt_list(
                ir, st.body, ctx, mod_parts
            )
            st.orelse = RewriteInlineCallsPass._rewrite_stmt_list(
                ir, st.orelse, ctx, mod_parts
            )
            return pro + [st]

        if isinstance(st, PyIRWhile):
            pro, test = RewriteInlineCallsPass._rewrite_expr(
                ir, st.test, ctx, mod_parts
            )
            st.test = test
            st.body = RewriteInlineCallsPass._rewrite_stmt_list(
                ir, st.body, ctx, mod_parts
            )
            return pro + [st]

        if isinstance(st, PyIRFor):
            pro_iter, it = RewriteInlineCallsPass._rewrite_expr(
                ir, st.iter, ctx, mod_parts
            )
            st.iter = it
            st.body = RewriteInlineCallsPass._rewrite_stmt_list(
                ir, st.body, ctx, mod_parts
            )
            return pro_iter + [st]

        if isinstance(st, PyIRWith):
            pro_with: List[PyIRNode] = []
            for item in st.items:
                p, e = RewriteInlineCallsPass._rewrite_expr(
                    ir, item.context_expr, ctx, mod_parts
                )
                pro_with.extend(p)
                item.context_expr = e

            st.body = RewriteInlineCallsPass._rewrite_stmt_list(
                ir, st.body, ctx, mod_parts
            )
            return pro_with + [st]

        if isinstance(st, PyIRFunctionDef):
            st.body = RewriteInlineCallsPass._rewrite_stmt_list(
                ir, st.body, ctx, mod_parts
            )
            return [st]

        if (
            isinstance(st, PyIRAssign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], PyIRVarUse)
        ):
            t0 = st.targets[0]

            if isinstance(st.value, PyIRCall):
                inl = RewriteInlineCallsPass._maybe_inline_assign_into_target(
                    ir, st.value, ctx, mod_parts=mod_parts, target=t0
                )
                if inl is not None:
                    return inl

            pro, v = RewriteInlineCallsPass._rewrite_expr(ir, st.value, ctx, mod_parts)
            st.value = v
            return pro + [st]

        if isinstance(st, PyIRAssign):
            pro, v = RewriteInlineCallsPass._rewrite_expr(ir, st.value, ctx, mod_parts)
            st.value = v
            return pro + [st]

        if isinstance(st, PyIRReturn):
            if st.value is None:
                return [st]
            pro, v = RewriteInlineCallsPass._rewrite_expr(ir, st.value, ctx, mod_parts)
            st.value = v
            return pro + [st]

        if isinstance(st, PyIRCall):
            inl = RewriteInlineCallsPass._maybe_inline_stmt(st, ctx, mod_parts)
            if inl is not None:
                return inl

            pro, expr = RewriteInlineCallsPass._rewrite_expr(ir, st, ctx, mod_parts)
            return pro + [expr]

        return [st]

    @staticmethod
    def _rewrite_expr(
        ir: PyIRFile,
        expr: PyIRNode,
        ctx: ExpandContext,
        mod_parts: tuple[str, ...] | None,
    ) -> Tuple[List[PyIRNode], PyIRNode]:
        if isinstance(expr, PyIRCall):
            return RewriteInlineCallsPass._maybe_inline_expr(ir, expr, ctx, mod_parts)

        if not is_dataclass(expr):
            return [], expr

        pro: List[PyIRNode] = []
        for f in fields(expr):
            v = getattr(expr, f.name)

            if isinstance(v, PyIRNode):
                p, nv = RewriteInlineCallsPass._rewrite_expr(ir, v, ctx, mod_parts)
                pro.extend(p)
                setattr(expr, f.name, nv)
                continue

            if isinstance(v, list):
                new_list: List[object] = []
                changed = False
                for x in v:
                    if isinstance(x, PyIRNode):
                        p, nx = RewriteInlineCallsPass._rewrite_expr(
                            ir, x, ctx, mod_parts
                        )
                        pro.extend(p)
                        new_list.append(nx)
                        changed = True
                    else:
                        new_list.append(x)
                if changed:
                    setattr(expr, f.name, new_list)
                continue

            if isinstance(v, dict):
                new_dict: Dict[str, object] = {}
                changed = False
                for k, x in v.items():
                    if isinstance(x, PyIRNode):
                        p, nx = RewriteInlineCallsPass._rewrite_expr(
                            ir, x, ctx, mod_parts
                        )
                        pro.extend(p)
                        new_dict[k] = nx
                        changed = True
                    else:
                        new_dict[k] = x
                if changed:
                    setattr(expr, f.name, new_dict)
                continue

        return pro, expr

    @staticmethod
    def _maybe_inline_stmt(
        call: PyIRCall,
        ctx: ExpandContext,
        mod_parts: tuple[str, ...] | None,
    ) -> List[PyIRNode] | None:
        key = _call_inline_key(mod_parts, call.func)
        if key is None:
            return None

        tmpl = ctx.inline_templates.get(key)
        if tmpl is None:
            return None

        block = RewriteInlineCallsPass._build_inline_do(call, ctx, tmpl, ret_store=None)
        return [block]

    @staticmethod
    def _maybe_inline_assign_into_target(
        ir: PyIRFile,
        call: PyIRCall,
        ctx: ExpandContext,
        *,
        mod_parts: tuple[str, ...] | None,
        target: PyIRVarUse,
    ) -> List[PyIRNode] | None:
        key = _call_inline_key(mod_parts, call.func)
        if key is None:
            return None

        tmpl = ctx.inline_templates.get(key)
        if tmpl is None:
            return None

        pro_args, new_call = RewriteInlineCallsPass._rewrite_call_args(
            ir, call, ctx, mod_parts
        )

        expr_inl = RewriteInlineCallsPass._try_inline_as_expr(new_call, tmpl)
        if expr_inl is not None:
            out: List[PyIRNode] = []
            out.extend(pro_args)
            out.append(_mk_assign(target, expr_inl, call))
            return out

        pre: List[PyIRNode] = []
        pre.extend(pro_args)
        pre.append(_mk_assign(target, _mk_nil(call), call))
        block = RewriteInlineCallsPass._build_inline_do(
            new_call, ctx, tmpl, ret_store=target
        )
        return pre + [block]

    @staticmethod
    def _maybe_inline_expr(
        ir: PyIRFile,
        call: PyIRCall,
        ctx: ExpandContext,
        mod_parts: tuple[str, ...] | None,
    ) -> Tuple[List[PyIRNode], PyIRNode]:
        key = _call_inline_key(mod_parts, call.func)
        if key is None:
            return [], call

        tmpl = ctx.inline_templates.get(key)
        if tmpl is None:
            return [], call

        pro_args, new_call = RewriteInlineCallsPass._rewrite_call_args(
            ir, call, ctx, mod_parts
        )
        expr_inl = RewriteInlineCallsPass._try_inline_as_expr(new_call, tmpl)
        if expr_inl is not None:
            return pro_args, expr_inl

        tmp = ctx.new_tmp_name()
        tmp_var = _mk_var(tmp, call)

        pre: List[PyIRNode] = []
        pre.extend(pro_args)
        pre.append(_mk_assign(tmp_var, _mk_nil(call), call))

        block = RewriteInlineCallsPass._build_inline_do(
            new_call, ctx, tmpl, ret_store=tmp_var
        )
        return pre + [block], tmp_var

    @staticmethod
    def _rewrite_call_args(
        ir: PyIRFile,
        call: PyIRCall,
        ctx: ExpandContext,
        mod_parts: tuple[str, ...] | None,
    ) -> Tuple[List[PyIRNode], PyIRCall]:
        pro: List[PyIRNode] = []
        new_call = deepcopy(call)

        new_args_p: List[PyIRNode] = []
        for a in new_call.args_p:
            p, na = RewriteInlineCallsPass._rewrite_expr(ir, a, ctx, mod_parts)
            pro.extend(p)
            new_args_p.append(na)
        new_call.args_p = new_args_p

        new_args_kw: Dict[str, PyIRNode] = {}
        for k, a in new_call.args_kw.items():
            p, na = RewriteInlineCallsPass._rewrite_expr(ir, a, ctx, mod_parts)
            pro.extend(p)
            new_args_kw[k] = na
        new_call.args_kw = new_args_kw

        return pro, new_call

    @staticmethod
    def _try_inline_as_expr(call: PyIRCall, tmpl: InlineTemplate) -> PyIRNode | None:
        """
        Inline only if template is effectively:
            return <expr>
        and duplicating args is safe (no duplicated calls).
        """
        body = _strip_nops(list(tmpl.fn.body))
        if len(body) != 1:
            return None
        if not isinstance(body[0], PyIRReturn):
            return None

        ret_expr = body[0].value

        subst: Dict[str, PyIRNode] = {}
        params = list(tmpl.params)

        for i, p in enumerate(params):
            arg: PyIRNode | None = None
            if i < len(call.args_p):
                arg = call.args_p[i]
            elif p in call.args_kw:
                arg = call.args_kw[p]
            else:
                sig = tmpl.fn.signature.get(p)
                default = sig[1] if sig is not None else None
                if default is not None:
                    arg = deepcopy(default)
                else:
                    return None

            subst[p] = arg

        uses: Dict[str, int] = {}
        if ret_expr is None:
            return None
        _count_var_uses(ret_expr, uses)

        for p, cnt in uses.items():
            if cnt <= 1:
                continue
            if p in subst and _has_call(subst[p]):
                return None

        return _subst_vars(ret_expr, subst)

    @staticmethod
    def _build_inline_do(
        call: PyIRCall,
        ctx: ExpandContext,
        tmpl: InlineTemplate,
        *,
        ret_store: PyIRNode | None,
    ) -> PyIRDo:
        prefix = ctx.new_inline_prefix()
        rename: Dict[str, str] = {name: f"{prefix}_{name}" for name in tmpl.locals}

        body = [_rename_vars(deepcopy(st), rename) for st in tmpl.fn.body]
        do_body: List[PyIRNode] = []

        param_set = set(tmpl.params)
        for name in tmpl.locals:
            if name in param_set:
                continue
            do_body.append(_mk_assign(_mk_var(rename[name], call), _mk_nil(call), call))

        args_p = call.args_p
        args_kw = call.args_kw

        for i, p in enumerate(tmpl.params):
            arg: PyIRNode | None = None
            if i < len(args_p):
                arg = args_p[i]
            elif p in args_kw:
                arg = args_kw[p]

            if arg is None:
                continue

            do_body.append(_mk_assign(_mk_var(rename[p], call), deepcopy(arg), call))

        if not _has_any_return(body):
            do_body.extend(body)
            return _mk_do(do_body, call)

        if _needs_label(body):
            end_label = f"{prefix}_end"
            do_body.extend(
                _rewrite_returns_with_label(
                    body,
                    ret_store=ret_store,
                    end_label=end_label,
                    ref=call,
                    top_level=True,
                )
            )
            do_body.append(_mk_label(end_label, call))
            return _mk_do(do_body, call)

        do_body.extend(_rewrite_returns_no_label(body, ret_store=ret_store, ref=call))
        return _mk_do(do_body, call)

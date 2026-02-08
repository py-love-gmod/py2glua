from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Callable, TypeVar, cast

from ..compiler_ir import PyIRFunctionExpr
from ...py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRDecorator,
    PyIREmitExpr,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRReturn,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)

_TNode = TypeVar("_TNode", bound=PyIRNode)


def _rewrite_typed_node_list(
    items: list[_TNode],
    *,
    rw_expr: Callable[[PyIRNode, bool], PyIRNode],
    store: bool,
) -> list[_TNode]:
    out: list[_TNode] = []
    for item in items:
        rewritten = rw_expr(item, store)
        if not isinstance(rewritten, type(item)):
            raise TypeError(
                f"Ожидался тип {type(item).__name__}, получен {type(rewritten).__name__}"
            )
        out.append(cast(_TNode, rewritten))
    return out


def rewrite_dataclass_children(
    node: PyIRNode,
    rw_node: Callable[[PyIRNode], PyIRNode],
) -> bool:
    if not is_dataclass(node):
        return False

    for f in fields(node):
        v = getattr(node, f.name)
        if isinstance(v, PyIRNode):
            setattr(node, f.name, rw_node(v))
            continue

        if isinstance(v, list):
            setattr(
                node,
                f.name,
                [rw_node(x) if isinstance(x, PyIRNode) else x for x in v],
            )
            continue

        if isinstance(v, dict):
            setattr(
                node,
                f.name,
                {
                    k: (rw_node(x) if isinstance(x, PyIRNode) else x)
                    for k, x in v.items()
                },
            )
            continue

        if isinstance(v, tuple):
            out: list[object] = []
            changed = False
            for x in v:
                if isinstance(x, PyIRNode):
                    nx = rw_node(x)
                    changed = changed or (nx is not x)
                    out.append(nx)
                else:
                    out.append(x)
            if changed:
                setattr(node, f.name, tuple(out))

    return True


def rewrite_stmt_block(
    body: list[PyIRNode],
    rw_stmt: Callable[[PyIRNode], list[PyIRNode]],
) -> list[PyIRNode]:
    out: list[PyIRNode] = []
    for st in body:
        out.extend(rw_stmt(st))
    return out


def rewrite_expr_with_store_default(
    node: PyIRNode,
    *,
    rw_expr: Callable[[PyIRNode, bool], PyIRNode],
    rw_block: Callable[[list[PyIRNode]], list[PyIRNode]],
) -> PyIRNode:
    """
    Common default walker for store-aware rewriters.

    It handles the most repeated IR node kinds and recursively delegates to
    `rw_expr(..., store=...)` and `rw_block(...)`.
    """
    if isinstance(node, PyIRAttribute):
        node.value = rw_expr(node.value, False)
        return node

    if isinstance(node, PyIRCall):
        node.func = rw_expr(node.func, False)
        node.args_p = [rw_expr(a, False) for a in node.args_p]
        node.args_kw = {k: rw_expr(v, False) for k, v in node.args_kw.items()}
        return node

    if isinstance(node, PyIRAssign):
        node.targets = [rw_expr(t, True) for t in node.targets]
        node.value = rw_expr(node.value, False)
        return node

    if isinstance(node, PyIRAugAssign):
        node.target = rw_expr(node.target, True)
        node.value = rw_expr(node.value, False)
        return node

    if isinstance(node, PyIRWithItem):
        node.context_expr = rw_expr(node.context_expr, False)
        if node.optional_vars is not None:
            node.optional_vars = rw_expr(node.optional_vars, True)
        return node

    if isinstance(node, PyIRWith):
        node.items = _rewrite_typed_node_list(
            node.items,
            rw_expr=rw_expr,
            store=False,
        )
        node.body = rw_block(node.body)
        return node

    if isinstance(node, PyIRIf):
        node.test = rw_expr(node.test, False)
        node.body = rw_block(node.body)
        node.orelse = rw_block(node.orelse)
        return node

    if isinstance(node, PyIRWhile):
        node.test = rw_expr(node.test, False)
        node.body = rw_block(node.body)
        return node

    if isinstance(node, PyIRFor):
        node.target = rw_expr(node.target, True)
        node.iter = rw_expr(node.iter, False)
        node.body = rw_block(node.body)
        return node

    if isinstance(node, PyIRReturn):
        if node.value is not None:
            node.value = rw_expr(node.value, False)
        return node

    if isinstance(node, PyIRDecorator):
        node.exper = rw_expr(node.exper, False)
        node.args_p = [rw_expr(a, False) for a in node.args_p]
        node.args_kw = {k: rw_expr(v, False) for k, v in node.args_kw.items()}
        return node

    if isinstance(node, PyIREmitExpr):
        node.args_p = [rw_expr(a, False) for a in node.args_p]
        node.args_kw = {k: rw_expr(v, False) for k, v in node.args_kw.items()}
        return node

    if isinstance(node, PyIRFunctionDef):
        node.decorators = _rewrite_typed_node_list(
            node.decorators,
            rw_expr=rw_expr,
            store=False,
        )
        new_func_sig: dict[str, tuple[str | None, PyIRNode | None]] = {}
        for k, (ann, default) in node.signature.items():
            if default is not None:
                default = rw_expr(default, False)
            new_func_sig[k] = (ann, default)
        node.signature = new_func_sig
        node.body = rw_block(node.body)
        return node

    if isinstance(node, PyIRFunctionExpr):
        new_expr_sig: dict[str, tuple[str | None, PyIRNode | None]] = {}
        for k, (ann, default) in node.signature.items():
            if default is not None:
                default = rw_expr(default, False)
            new_expr_sig[k] = (ann, default)
        node.signature = new_expr_sig
        node.body = rw_block(node.body)
        return node

    if isinstance(node, PyIRClassDef):
        node.decorators = _rewrite_typed_node_list(
            node.decorators,
            rw_expr=rw_expr,
            store=False,
        )
        node.bases = [rw_expr(b, False) for b in node.bases]
        node.body = rw_block(node.body)
        return node

    rewrite_dataclass_children(node, lambda x: rw_expr(x, False))
    return node


def replace_or_prepend_init_file(
    files: list[PyIRFile],
    init_ir: PyIRFile,
    *,
    init_path: str,
) -> list[PyIRFile]:
    out: list[PyIRFile] = [init_ir]
    for ir in files:
        if ir.path is not None and ir.path.as_posix() == init_path:
            continue
        out.append(ir)
    return out

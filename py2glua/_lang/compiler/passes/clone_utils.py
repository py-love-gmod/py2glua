from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Dict

from ...py.ir_dataclass import PyIRNode, PyIRVarUse
from .analysis.symlinks import PyIRSymLink


def clone_deep(node: PyIRNode) -> PyIRNode:
    def clone_value(v):
        if isinstance(v, PyIRNode):
            return clone_node(v)

        if isinstance(v, list):
            return [clone_value(x) for x in v]

        if isinstance(v, dict):
            return {k: clone_value(x) for k, x in v.items()}

        if isinstance(v, tuple):
            return tuple(clone_value(x) for x in v)

        return v

    def clone_node(n: PyIRNode) -> PyIRNode:
        if not is_dataclass(n):
            return n

        cls = n.__class__
        kwargs = {f.name: clone_value(getattr(n, f.name)) for f in fields(n)}
        return cls(**kwargs)  # type: ignore

    return clone_node(node)


def clone_with_pos(
    node: PyIRNode,
    *,
    line: int | None,
    offset: int | None,
) -> PyIRNode:
    def clone_value(v):
        if isinstance(v, PyIRNode):
            return clone_node(v)

        if isinstance(v, list):
            return [clone_value(x) for x in v]

        if isinstance(v, dict):
            return {k: clone_value(x) for k, x in v.items()}

        if isinstance(v, tuple):
            return tuple(clone_value(x) for x in v)

        return v

    def clone_node(n: PyIRNode) -> PyIRNode:
        if not is_dataclass(n):
            return n

        cls = n.__class__
        kwargs = {}

        for f in fields(n):
            if f.name == "line":
                kwargs[f.name] = line
            elif f.name == "offset":
                kwargs[f.name] = offset
            else:
                kwargs[f.name] = clone_value(getattr(n, f.name))

        return cls(**kwargs)

    return clone_node(node)


def clone_with_subst(
    node: PyIRNode,
    subst: Dict[str, PyIRNode],
) -> PyIRNode:
    def force_load(n: PyIRNode) -> PyIRNode:
        if isinstance(n, PyIRSymLink) and n.is_store:
            return PyIRSymLink(
                line=n.line,
                offset=n.offset,
                name=n.name,
                symbol_id=n.symbol_id,
                hops=n.hops,
                is_store=False,
                decl_file=n.decl_file,
                decl_line=n.decl_line,
            )
        return n

    def clone_value(v, *, apply_subst: bool):
        if isinstance(v, PyIRNode):
            return clone_node(v, apply_subst=apply_subst)

        if isinstance(v, list):
            return [clone_value(x, apply_subst=apply_subst) for x in v]

        if isinstance(v, dict):
            return {k: clone_value(x, apply_subst=apply_subst) for k, x in v.items()}

        return v

    def clone_node(n: PyIRNode, *, apply_subst: bool) -> PyIRNode:
        if apply_subst:
            if isinstance(n, PyIRVarUse) and n.name in subst:
                return clone_node(subst[n.name], apply_subst=False)

            if isinstance(n, PyIRSymLink) and n.name in subst:
                return clone_node(subst[n.name], apply_subst=False)

        if not is_dataclass(n):
            return n

        cls = n.__class__
        kwargs = {}

        for f in fields(n):
            value = getattr(n, f.name)

            if f.name in ("line", "offset"):
                kwargs[f.name] = value
            else:
                kwargs[f.name] = clone_value(value, apply_subst=apply_subst)

        out = cls(**kwargs)
        return force_load(out)

    return clone_node(node, apply_subst=True)

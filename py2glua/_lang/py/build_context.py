import tokenize
from collections.abc import Callable

from .ir_dataclass import PyIRNode

_build_block: Callable | None = None
_build_expr: Callable[[list[tokenize.TokenInfo]], PyIRNode] | None = None


def set_build_block(fn: Callable) -> None:
    global _build_block
    _build_block = fn


def build_block(nodes):
    if _build_block is None:
        raise RuntimeError("build_block is not initialized")

    return _build_block(nodes)


def set_build_expr(fn: Callable[[list[tokenize.TokenInfo]], PyIRNode]) -> None:
    global _build_expr
    _build_expr = fn


def build_expr(tokens: list[tokenize.TokenInfo]) -> PyIRNode:
    if _build_expr is None:
        raise RuntimeError("build_expr is not initialized")

    return _build_expr(tokens)

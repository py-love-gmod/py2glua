from collections.abc import Callable

_build_block: Callable | None = None


def set_build_block(fn: Callable) -> None:
    global _build_block
    _build_block = fn


def build_block(nodes):
    if _build_block is None:
        raise RuntimeError("build_block is not initialized")

    return _build_block(nodes)

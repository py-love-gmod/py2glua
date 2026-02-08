from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, Callable

Fore: Any
Style: Any
init: Callable[..., None]

try:
    _colorama = importlib.import_module("colorama")
    Fore = getattr(_colorama, "Fore")
    Style = getattr(_colorama, "Style")
    init = getattr(_colorama, "init")
except Exception:
    Fore = SimpleNamespace(
        CYAN="",
        GREEN="",
        YELLOW="",
        RED="",
        MAGENTA="",
    )
    Style = SimpleNamespace(
        BRIGHT="",
        RESET_ALL="",
    )

    def init(*args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

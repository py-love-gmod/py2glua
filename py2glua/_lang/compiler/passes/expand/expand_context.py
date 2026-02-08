from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ....py.ir_dataclass import PyIRFunctionDef, PyIRNode
from ...compiler_ir import PyIRFunctionExpr


@dataclass(frozen=True)
class ContextManagerTemplate:
    pre: Tuple[PyIRNode, ...]
    post: Tuple[PyIRNode, ...]
    insert_with_body: bool
    params: Tuple[str, ...]


@dataclass(frozen=True)
class FnSig:
    params: Tuple[str, ...]
    defaults: Dict[str, PyIRNode]
    vararg_name: str | None = None
    kwarg_name: str | None = None


@dataclass(frozen=True)
class ClassTemplate:
    instance_methods: Dict[str, FnSig]
    class_methods: Tuple[str, ...]
    has_init: bool


@dataclass(frozen=True)
class InlineTemplate:
    fn: PyIRFunctionDef
    params: Tuple[str, ...]
    locals: Tuple[str, ...]


@dataclass
class ExpandContext:
    cm_templates: Dict[str, ContextManagerTemplate] = field(default_factory=dict)
    # Per-file top-level function signatures used by args normalization.
    # Key is canonical file path string.
    fn_sigs: Dict[str, Dict[str, FnSig]] = field(default_factory=dict)
    # Per-file class runtime templates.
    # Key is canonical file path string.
    class_templates: Dict[str, Dict[str, ClassTemplate]] = field(default_factory=dict)

    inline_templates: Dict[str, InlineTemplate] = field(default_factory=dict)

    anonymous_templates: Dict[str, PyIRFunctionExpr] = field(default_factory=dict)

    tmp_counter: int = 0

    def new_tmp_name(self) -> str:
        self.tmp_counter += 1
        return f"__tmp_{self.tmp_counter}"

    def new_inline_prefix(self) -> str:
        self.tmp_counter += 1
        return f"__inl_{self.tmp_counter}"

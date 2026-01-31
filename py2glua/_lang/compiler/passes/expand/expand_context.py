from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ....py.ir_dataclass import PyIRNode


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


@dataclass
class ExpandContext:
    cm_templates: Dict[str, ContextManagerTemplate] = field(default_factory=dict)
    fn_sigs: Dict[str, FnSig] = field(default_factory=dict)

    tmp_counter: int = 0

    def new_tmp_name(self) -> str:
        self.tmp_counter += 1
        return f"__tmp_{self.tmp_counter}"

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .types import Type


@dataclass
class InferenceResult:
    node_types: Dict[int, Type] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

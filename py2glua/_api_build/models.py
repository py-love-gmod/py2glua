from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BODY_PASS = "pass"
BODY_ELLIPSIS = "ellipsis"
BODY_DECORATOR = "decorator"
BODY_RAW = "raw"
ALLOWED_BODIES = {BODY_PASS, BODY_ELLIPSIS, BODY_DECORATOR, BODY_RAW}


@dataclass(slots=True)
class ApiArgSpec:
    name: str
    ann: str | None
    kind: str = "normal"
    default: Any = None
    has_default: bool = False


@dataclass(slots=True)
class ApiDecoratorSpec:
    kind: str
    options: dict[str, Any]


@dataclass(slots=True)
class ApiTypeVarSpec:
    name: str
    expression: str


@dataclass(slots=True)
class ApiFunctionSpec:
    name: str
    args: list[ApiArgSpec]
    returns: str | None
    body: str
    decorators: list[ApiDecoratorSpec]
    body_lines: list[str] = field(default_factory=list)
    docstring: str | None = None
    deprecated: str | None = None


@dataclass(slots=True)
class ApiFieldSpec:
    name: str
    ann: str | None
    has_value: bool
    value: Any = None
    docstring: str | None = None


@dataclass(slots=True)
class ApiClassIfSpec:
    condition: str
    fields: list[ApiFieldSpec]
    methods: list[ApiFunctionSpec]
    classes: list["ApiClassSpec"]


@dataclass(slots=True)
class ApiClassSpec:
    name: str
    bases: list[str]
    decorators: list[ApiDecoratorSpec]
    fields: list[ApiFieldSpec]
    methods: list[ApiFunctionSpec]
    classes: list["ApiClassSpec"]
    ifs: list[ApiClassIfSpec] = field(default_factory=list)
    docstring: str | None = None
    deprecated: str | None = None

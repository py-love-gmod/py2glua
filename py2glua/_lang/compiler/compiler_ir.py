from __future__ import annotations

from dataclasses import dataclass, field

from ..py.ir_dataclass import PyIRNode


@dataclass
class PyIRDo(PyIRNode):
    """
    Lua scope block:
        do
            ...
        end
    """

    body: list["PyIRNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for n in self.body:
            yield from n.walk()


@dataclass
class PyIRLabel(PyIRNode):
    """
    Lua label:
        ::name::
    """

    name: str

    def walk(self):
        yield self


@dataclass
class PyIRGoto(PyIRNode):
    """
    Lua goto:
        goto name
    """

    label: str

    def walk(self):
        yield self


@dataclass
class PyIRFunctionExpr(PyIRNode):
    """
    Lua function literal (expression):
        function(args) ... end

    signature: same shape as PyIRFunctionDef.signature:
      dict[str, tuple[str|None, PyIRNode|None]]
    """

    signature: dict[str, tuple[str | None, PyIRNode | None]]
    vararg: str | None = None
    kwarg: str | None = None
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self

        for _ann, default in self.signature.values():
            if default is not None:
                yield from default.walk()

        for n in self.body:
            yield from n.walk()

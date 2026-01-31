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

    signature: как у PyIRFunctionDef.signature
    """

    signature: dict[str, tuple[str | None, PyIRNode | None]]
    body: list[PyIRNode] = field(default_factory=list)

    def walk(self):
        yield self
        for n in self.body:
            yield from n.walk()

from dataclasses import dataclass

from ..py.ir_dataclass import PyIRNode


@dataclass
class PyIRSymbolRef(PyIRNode):
    """
    Семантическая ссылка на символ по ID.
    Contract:
      - создаётся только на этапе анализа/линковки
      - emitter/нижележащие фазы должны уметь восстановить fqname через ctx
    """

    sym_id: int

    def walk(self):
        yield self

    def __str__(self) -> str:
        return f"<SYM#{self.sym_id}>"


@dataclass
class PyIRLocalRef(PyIRNode):
    local_id: int
    name: str

    def walk(self):
        yield self

    def __str__(self) -> str:
        return f"<L_SYM#{self.local_id}>"

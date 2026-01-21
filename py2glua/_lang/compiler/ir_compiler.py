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

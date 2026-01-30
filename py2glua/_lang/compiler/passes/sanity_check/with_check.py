from __future__ import annotations

from ....._cli import CompilerExit
from ....py.ir_dataclass import PyIRFile, PyIRWithItem


class WithSanityCheckPass:
    """
    Запрещает использование `as` в инструкции with.
    """

    @staticmethod
    def run(ir: PyIRFile) -> None:
        for node in ir.walk():
            if not isinstance(node, PyIRWithItem):
                continue

            if node.optional_vars is not None:
                CompilerExit.user_error_node(
                    "Использование 'as' в инструкции with запрещено",
                    ir.path,
                    node,
                )

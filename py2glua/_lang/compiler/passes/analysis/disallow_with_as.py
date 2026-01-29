from __future__ import annotations

from ....._cli import CompilerExit
from ....py.ir_dataclass import PyIRFile, PyIRWith


class DisallowWithAsPass:
    """
    Запрещает использование конструкции:

        with expr as name:
    """

    _CTX_SEEN = "_withas_seen_paths"
    _CTX_DONE = "_withas_done"

    @staticmethod
    def run(ir: PyIRFile, *args) -> None:
        for node in ir.walk():
            if not isinstance(node, PyIRWith):
                continue

            for item in node.items:
                if item.optional_vars is not None:
                    CompilerExit.user_error_node(
                        "Конструкция 'with ... as имя' запрещена.",
                        ir.path,
                        node,
                    )

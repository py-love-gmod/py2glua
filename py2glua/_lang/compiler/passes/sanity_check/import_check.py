from __future__ import annotations

from ....._cli import CompilerExit
from ....py.ir_dataclass import PyIRFile, PyIRImport


class ImportSanityCheckPass:
    @staticmethod
    def run(ir: PyIRFile) -> None:
        top_level_imports = list(ir.body)

        for node in ir.walk():
            if not isinstance(node, PyIRImport):
                continue

            if node not in top_level_imports:
                CompilerExit.user_error_node(
                    "Инструкции import запрещены внутри функций, условий и других блоков",
                    ir.path,
                    node,
                )

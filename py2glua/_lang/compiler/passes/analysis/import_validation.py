from ....._cli import CompilerExit
from ....py.ir_dataclass import PyIRFile, PyIRImport


class ImportValidationPass:
    @staticmethod
    def run(ir: PyIRFile, *args) -> None:
        top_level_imports = [node for node in ir.body if isinstance(node, PyIRImport)]

        for node in ir.walk():
            if not isinstance(node, PyIRImport):
                continue

            if node not in top_level_imports:
                CompilerExit.user_error_node(
                    "Импорт разрешён только на уровне файла",
                    path=ir.path,
                    node=node,
                )

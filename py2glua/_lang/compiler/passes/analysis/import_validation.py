from ....._cli.logging_setup import exit_with_code
from ....py.ir_dataclass import PyIRFile, PyIRImport


class ImportValidationPass:
    @staticmethod
    def run(ir: PyIRFile, *args) -> None:
        top_level_imports = [node for node in ir.body if isinstance(node, PyIRImport)]

        for node in ir.walk():
            if not isinstance(node, PyIRImport):
                continue

            if node not in top_level_imports:
                exit_with_code(
                    1,
                    f"Импорт разрешён только на уровне файла\nФайл: {ir.path}\nLINE|OFFSET: {node.line}|{node.offset}",
                )

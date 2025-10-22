from ..ir.ir_base import File, Import, ImportType, IRNode
from .base_normalizer import BaseNormalizer


class ImportNormalizer(BaseNormalizer):
    priority = 1

    @classmethod
    def run(cls, file: File) -> None:
        for node in file.walk():
            if node is file:
                continue

            if isinstance(node, Import):
                cls._normilize_import(node, file)

            else:
                cls._normilize_nodes(node, file)

        del file.meta["import_map"]

    @classmethod
    def _set_import_type(cls, node: Import):
        ImportType.EXTERNAL  # Импорты pip
        ImportType.INTERNAL  # Импорты py2glua
        ImportType.LOCAL  # Локальные импорты
        ImportType.PYTHON_STD  # Питоновские стандартные

    @classmethod
    def _normilize_import(cls, node: Import, file: File):
        file.meta["import_map"] = {}

        # TODO: Сделать маппинг в мету
        cls._set_import_type(node)
        # TODO: Сделать нормализацию
        # from x import y -> import x.y
        # from x import y as z -> import x.y

    @classmethod
    def _normilize_nodes(cls, node: IRNode, file: File):
        # TODO: Сверяемся с маппингом файла и меняем
        # y.func() -> x.y.func()
        # z.func() -> x.y.func()
        pass

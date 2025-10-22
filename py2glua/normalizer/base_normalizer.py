from ..ir.ir_base import File


class BaseNormalizer:
    priority: int = 0

    @classmethod
    def run(cls, file: File) -> None:
        raise NotImplementedError

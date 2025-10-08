from ast import AST


class BaseConvertor:
    """Интерфейс для базового конвертора в glua код"""

    @classmethod
    def convert(cls, node: AST, intend: int = 0) -> str:
        """Перевод ноды в glua код

        Args:
            node (AST): Нода
            intend (int, optional): Отступ. Defaults to 0.

        Returns:
            str: Glua код
        """
        raise NotImplementedError

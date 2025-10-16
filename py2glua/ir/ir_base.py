"""
Блок классов необходимых для корректной работы транслятора. Базовое абстрактное дерево питона
парсить и работать ощущается как хуйня. Легче сделать свою промежуточную структуру, чтобы уже её и оптимизировать,
и менять.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from ..runtime import Realm


@dataclass
class Node:
    """Базовый класс представления ноды промежуточных преобразований"""

    lineno: int | None  # Номер линии где находится нода
    col_offset: int | None  # Её начало относительно начала строки

    def walk(self) -> Iterator["Node"]:
        """Базовый метод позволяющий обходить рекурсивно все ноды"""
        raise NotImplementedError


@dataclass
class File(Node):
    """Класс отвечающий за представление единичного файла"""

    path: Path | None = None  # Путь до файла
    realm: Realm | None = None  # Область выполнения кода файла
    meta_tags: list[str] = field(default_factory=list)  # Мета теги кода
    body: list[Node] = field(default_factory=list)  # Сам код файла

    def walk(self) -> Iterator[Node]:
        yield self
        for ch in self.body:
            yield ch


# region
class ImportType(Enum):
    UNKNOWN = auto()
    PYTHON_STD = auto()  # стандартная библиотека
    EXTERNAL = auto()  # сторонние пакеты (из site-packages)
    LOCAL = auto()  # относительный импорт внутри проекта
    INTERNAL = auto()  # собственные IR/Runtime-модули


@dataclass
class Import(Node):
    """Представление импорта питоновского файла"""

    module: str | None = None
    names: list[str] = field(default_factory=list)
    aliases: list[str | None] = field(default_factory=list)
    import_type: ImportType = ImportType.UNKNOWN

    def walk(self) -> Iterator[Node]:
        yield self


# endregion
@dataclass
class Constant(Node):
    """Класс отвечающий за представление констант"""

    value: int | float | str | bool | None

    def walk(self) -> Iterator[Node]:
        yield self


# region vars
@dataclass
class VarStore(Node):
    """Класс сохранения переменной"""

    name: str
    value: Node

    def walk(self) -> Iterator[Node]:
        yield self
        yield from self.value.walk()


@dataclass
class VarLoad(Node):
    """Класс загрузки переменной"""

    name: str

    def walk(self) -> Iterator[Node]:
        yield self


# endregion

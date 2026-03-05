from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, TypeVar

from ..realm import Realm
from .compiler_directive import CompilerDirective

_K = TypeVar("_K")
_V = TypeVar("_V")
_T = TypeVar("_T")


@CompilerDirective.no_compile()
class nil_type:
    """
    Glua-тип отсутствующего значения.
    - `None` означает "пустой объект", то есть значение существует.
    - `nil` в glua означает, что значения нет вообще.

    Переменная, которой присвоен `nil`, считается неопределённой.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "nil"

    def __str__(self) -> str:
        return "nil"

    def __bool__(self) -> bool:
        return False


# Runtime singleton for typing/user-facing nil marker.
nil: Final[nil_type] = nil_type()


@CompilerDirective.no_compile()
class lua_table:
    """
    Модель Lua-таблицы для Python-стороны.

    `array` хранит последовательную часть (индексы 1..N),
    `map` хранит произвольные ключи.
    """

    __slots__ = ("array", "map")

    def __init__(
        self,
        array: list[Any] | None = None,
        map: dict[Any, Any] | None = None,
    ) -> None:
        self.array = []
        if array is not None:
            self.array = array

        self.map = {}
        if map is not None:
            self.map = map

    @staticmethod
    @CompilerDirective.gmod_api("pairs", realm=[Realm.SHARED], method=False)
    def pairs(table_value: Mapping[_K, _V]) -> Iterable[tuple[_K, _V]]:
        """
        Итерация по всем элементам таблицы.

        Эквивалент Lua-функции `pairs`.

        Порядок обхода элементов не гарантируется и зависит
        от внутреннего состояния таблицы.

        Returns:
            Iterable[(key, value)]
        """
        ...

    @staticmethod
    @CompilerDirective.gmod_api("SortedPairs", realm=[Realm.SHARED], method=False)
    def SortedPairs(
        table_value: Mapping[_K, _V],
        descending: bool = False,
    ) -> Iterable[tuple[_K, _V]]:
        """
        Итерация по таблице с сортировкой по ключам.

        Аналог `pairs`, однако элементы возвращаются в
        детерминированном порядке, отсортированном по ключам.

        Args:
            table_value: таблица для обхода
            descending: если True - сортировка выполняется в обратном порядке

        Returns:
            Iterable[(key, value)]
        """
        ...

    @staticmethod
    @CompilerDirective.gmod_api(
        "SortedPairsByValue",
        realm=[Realm.SHARED],
        method=False,
    )
    def SortedPairsByValue(
        table_value: Mapping[_K, _V],
        descending: bool = False,
    ) -> Iterable[tuple[_K, _V]]:
        """
        Итерация по таблице с сортировкой по значениям.

        Работает аналогично `pairs`, но элементы предварительно
        сортируются по значению.

        Args:
            table_value: таблица для обхода
            descending: если True - сортировка выполняется в обратном порядке

        Returns:
            Iterable[(key, value)]
        """
        ...

    @staticmethod
    @CompilerDirective.gmod_api("RandomPairs", realm=[Realm.SHARED], method=False)
    def RandomPairs(table_value: Mapping[_K, _V]) -> Iterable[tuple[_K, _V]]:
        """
        Итерация по таблице в случайном порядке.

        Эквивалент Lua-функции `RandomPairs`.

        Используется, когда требуется намеренно недетерминированный
        порядок обхода элементов.

        Returns:
            Iterable[(key, value)]
        """
        ...

    @staticmethod
    @CompilerDirective.gmod_api("ipairs", realm=[Realm.SHARED], method=False)
    def ipairs(table_value: Sequence[_T]) -> Iterable[tuple[int, _T]]:
        """
        Итерация по последовательной части таблицы.

        Эквивалент Lua-функции `ipairs`. Обход выполняется
        по индексам `1..N` до первого отсутствующего значения.

        Returns:
            Iterable[(index, value)]
        """
        ...

    @staticmethod
    def by_len(table_value: Sequence[_T]) -> Iterable[_T]:
        """
        Итерация по индексам `1..#table`.

        В отличие от `ipairs`, использует длину таблицы (`#table`)
        как верхнюю границу диапазона.

        Этот метод используется компилятором как явная стратегия
        итерации при конструкции:

            for x in lua_table.by_len(table)

        Notes:
            Специальный метод компилятора.
        """
        ...

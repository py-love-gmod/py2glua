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
        TODO: NORMAL DOCS
        Явный выбор итерации через `pairs`.
        """
        ...

    @staticmethod
    @CompilerDirective.gmod_api("ipairs", realm=[Realm.SHARED], method=False)
    def ipairs(table_value: Sequence[_T]) -> Iterable[tuple[int, _T]]:
        """
        TODO: NORMAL DOCS
        Явный выбор итерации через `ipairs`.
        """
        ...

    @staticmethod
    def by_len(table_value: Sequence[_T]) -> Iterable[_T]:
        """
        TODO: NORMAL DOCS
        WARN: MAGIC METHOD
        Явная стратегия итерации через индекс `1..#table`.

        Используется компилятором в `for ... in lua_table.by_len(table)`.
        """
        ...

    @staticmethod
    @CompilerDirective.gmod_api("RandomPairs", realm=[Realm.SHARED], method=False)
    def random_pairs(table_value: Mapping[_K, _V]) -> Iterable[tuple[_K, _V]]:
        """
        TODO: NORMAL DOCS
        Явный выбор итерации через `RandomPairs`.
        """
        ...

    @staticmethod
    @CompilerDirective.gmod_api("SortedPairs", realm=[Realm.SHARED], method=False)
    def sorted_pairs(table_value: Mapping[_K, _V]) -> Iterable[tuple[_K, _V]]:
        """
        TODO: NORMAL DOCS
        Явный выбор итерации через `SortedPairs`.
        """
        ...

    @staticmethod
    @CompilerDirective.gmod_api(
        "SortedPairsByValue",
        realm=[Realm.SHARED],
        method=False,
    )
    def sorted_pairs_by_value(table_value: Mapping[_K, _V]) -> Iterable[tuple[_K, _V]]:
        """
        TODO: NORMAL DOCS
        Явный выбор итерации через `SortedPairsByValue`.
        """
        ...

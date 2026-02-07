from collections.abc import Callable
from enum import Enum

from .core import CompilerDirective, nil


class ENTType(Enum):
    nextbot = "nextbot"
    """A NextBot NPC. A newer-type NPCs with better navigation.

    All NextBot functions and hooks are also usable on these entities."""

    anim = "anim"
    """A normal entity with visual and/or physical presence in the game world, such as props or whatever else you can imagine."""

    brush = "brush"
    """A serverside only trigger entity. Mostly used very closely with the Hammer Level Editor."""

    point = "point"
    """A usually serverside only entity that doesn't have a visual or physical representation in the game world, such as logic entities."""

    filter = "filter"
    """A different kind of "point" entity used in conjunction with trigger (brush type) entities."""

    ai = "ai"
    """2004 Source Engine NPC system entity"""


@CompilerDirective.gmod_prototype("entities")
class ENT:
    # Блок "шорткат" полей. Они не существуют в обычном lua
    # Но мне удобнее их добавить чтобы работать через них
    uid: str | nil
    """Указание ID энтити.
    По умолчанию идёт значение имени переменной
    """

    model: str = "models/hunter/blocks/cube025x025x025.mdl"
    """Объект модели"""

    # Обычные луа поля
    Type: ENTType = ENTType.anim
    """TODO:"""

    Base: str = "base_anim"
    """TODO:"""

    @staticmethod
    def override(
        realm: CompilerDirective.RealmMarker,
        target: Callable,
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator

    @staticmethod
    def add_method(
        realm: CompilerDirective.RealmMarker,
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator

    def Initialize(self) -> None:
        pass

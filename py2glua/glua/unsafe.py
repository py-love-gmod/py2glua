from pathlib import Path
from typing import Any
from warnings import deprecated

from .directive_compiler import InternalCompilerDirective
from .realm import Realm


@InternalCompilerDirective.stub()
class Unsafe:
    """Класс с небезопасными функциями, или функциями которые будут удалены в будующем. Все методы и декораторы из этого блока не советуется использовтаь от слова совсем."""

    @staticmethod
    def raw(lua_code: str) -> None:
        """Позволяет сказать компилятору "вставь голый луа сюдa". Что либо возвращать из этого блока нельзя"""
        pass

    @staticmethod
    @deprecated(
        "Данный метод остался лишь как отладочный для введения ресурсов вручную. В скором времени py2glua будет собирать все ресурсы сам через plg-sdk."
    )
    @InternalCompilerDirective.gmod_api(
        "AddCSLuaFile",
        [Realm.CLIENT, Realm.SERVER],
    )
    def AddCSLuaFile(file_path: Path) -> None:
        """Метод заставляет клиента загрузить файл который указан."""
        pass

    @staticmethod
    @deprecated(
        "Данный метод остался лишь как отладочный для введения ресурсов вручную.\nБинарные модули в скором времени будут зайдействовать отдельный API.\nВ скором времени py2glua будет собирать все ресурсы сам через plg-sdk."
    )
    @InternalCompilerDirective.gmod_api(
        "include",
        [Realm.CLIENT, Realm.SERVER],
    )
    def include(file_path: Path) -> Any:
        """Запускает файл по указаному пути."""
        pass

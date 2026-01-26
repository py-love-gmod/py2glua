from typing import Final

from .core import CompilerDirective


@CompilerDirective.gmod_special_enum(
    fields={
        "SERVER": ("global", "SERVER"),
        "CLIENT": ("global", "CLIENT"),
        "MENU": ("global", "MENU_DLL"),
        "SHARED": ("bool", True),
    }
)
@CompilerDirective.with_condition()
class Realm:
    """Среда выполнения кода. Данный класс можно использовать как и для определения среды всего файла, так и для проверки среды в рантайме."""

    SERVER: Final = CompilerDirective.RealmMarker()
    """Код выполняется исключительно на сервере"""

    CLIENT: Final = CompilerDirective.RealmMarker()
    """Код выполняется исключительно на клиенте"""

    MENU: Final = CompilerDirective.RealmMarker()
    """
    Отдельное Lua-состояние главного меню.
    Изолировано от CLIENT и SERVER, прямое взаимодействие невозможно.
    Связь возможна только через файловую систему.
    Требует перезаписи Lua-файлов движка.
    """

    SHARED: Final = CompilerDirective.RealmMarker()
    """Код выполняется одновременно и на клиенте, и на сервере"""

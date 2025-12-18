from .directive_compiler import InternalCompilerDirective


@InternalCompilerDirective.gmod_special_enum(
    fields={
        "SERVER": ("global", "SERVER"),
        "CLIENT": ("global", "CLIENT"),
        "SHARED": ("bool", True),
    }
)
class Realm:
    """Среда выполнения кода. Данный класс можно использовать как и для определения среды всего файла, так и для проверки среды в рантайме."""

    SERVER: bool
    """Код выполняется исключительно на сервере"""

    CLIENT: bool
    """Код выполняется исключительно на клиенте"""

    SHARED: bool
    """Код выполняется одновременно и на клиенте, и на сервере"""

from .directive_compiler import CompilerDirective


@CompilerDirective.internal.no_compile()
class nil:
    """
    glua-тип отсутствующего значения.
    - `None` означает "пустой объект", то есть значение существует.
    - `nil` в glua означает, что значения нет вообще.

    Переменная, которой присвоен `nil`, считается неопределённой.
    """

    __slots__ = ()

    def __repr__(self):
        return "nil"

    def __str__(self):
        return "nil"

    def __bool__(self):
        return False

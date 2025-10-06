class CompileError(Exception):
    """Ошибка компиляции кода из python в GLua"""

    pass


class NameError(Exception):
    """Ошибка наименования переменной в коде"""

    pass

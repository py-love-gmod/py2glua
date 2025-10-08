from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Global:
    """Вспомогательный класс для пометки функций и переменных как глобальных при генерации Lua-кода."""

    __slots__ = ()

    @classmethod
    def func(cls, func: Callable) -> Callable:
        """
        Используется как декоратор: ``@Global.func``.
        Помечает функцию как глобальную, чтобы в Lua она не объявлялась через `local`.
        """
        setattr(func, "__glua_global__", True)
        return func

    @staticmethod
    def mark_var(value: Any):
        """
        Используется для переменных: ``x = Global.mark_var(10)``.
        Возвращает значение без изменений, но помечает его как глобальное (если возможно).
        """
        try:
            setattr(value, "__glua_global__", True)
        except AttributeError:
            # Примитивные типы (int, str и т.п.) не поддерживают установку атрибутов, но мы всё равно считаем их глобальными маркерами.
            pass

        return value

    @staticmethod
    def is_global_annotation(node) -> bool:
        """Возвращает True, если узел аннотации представляет собой идентификатор `Global`."""
        import ast

        return isinstance(node, ast.Name) and node.id == "Global"

    @classmethod
    def is_func_decorator(cls, node) -> bool:
        """
        Проверяет, является ли узел декоратором ``@Global.func``.
        Поддерживает как обычный вариант (``@Global.func``), так и вызов (``@Global.func()``).
        """
        import ast

        # Прямое обращение Global.func
        if cls._is_global_attribute(node, "func"):
            return True

        # Вызов Global.func(...)
        if isinstance(node, ast.Call) and cls._is_global_attribute(node.func, "func"):
            return True

        return False

    @classmethod
    def unwrap_mark_var_call(cls, node):
        """
        Проверяет, является ли узел вызовом ``Global.mark_var(...)``.
        Возвращает кортеж ``(is_global_call, value_node)``, где:
          - is_global_call — True, если это действительно вызов mark_var;
          - value_node — первый позиционный аргумент или None, если аргумент отсутствует.
        """
        import ast

        if isinstance(node, ast.Call) and cls._is_global_attribute(
            node.func, "mark_var"
        ):
            if node.args:
                return True, node.args[0]

            for kw in node.keywords:
                if kw.arg == "value":
                    return True, kw.value

            return True, None

        return False, None

    @staticmethod
    def _is_global_attribute(node, attr: str) -> bool:
        """
        Проверяет, что узел представляет обращение вида ``Global.<attr>``.
        Например, Global.func или Global.mark_var.
        """
        import ast

        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "Global"
            and node.attr == attr
        )

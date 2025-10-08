from collections.abc import Callable
from typing import Any


class Global:
    """Класс маркер. Используется для пометки объектов как глобальных

    Для регистрации глобальных объектов используются следующие конструкции:

    * Для функций и классов используется декоратор `@Global.callable`
    * Для переменных используется синтаксическая конструкция `x = Global.var(some_value)`
    """

    _global_meta_attr = "__py2glua_global__"

    @classmethod
    def callable(cls, obj: Callable) -> Callable:
        """Декоратор использующийся обозначения класса или функции как глобальной

        Пример:
            ```python
            @Global.callable
            class SomeClass:
                pass

            @Global.callable
            def SomeFunc():
                pass
            ```

        Args:
            obj (Callable): Класс или функция

        Returns:
            Callable: Класс или функция с добавление глобального атрибута
        """
        setattr(obj, cls._global_meta_attr, True)
        return obj

    @classmethod
    def var(cls, value: Any) -> Any:
        """Метод использующийся для сообщению компилятора что переменная является глобальной

        Пример:
            ```python
            x = Global.var(some_value)
            ```

        Args:
            value (Any): Значение переменной

        Returns:
            Any: Значение переменной с изменённым атрибутом
        """
        try:
            setattr(value, cls._global_meta_attr, True)

        except Exception:
            # Базовые типы не поддерживают установку атрибутов
            # Фактически просто нужно автоматом инлайнить как не глобальную
            # TODO: Подумать как в ast сделать метадату для базовых типов
            pass

        return value

    @classmethod
    def is_global(cls, obj: Any) -> bool:
        """Возвращает является ли объект глобальным

        Args:
            obj (Any): Объект проверки

        Returns:
            bool: Является ли глобальным
        """
        return getattr(obj, cls._global_meta_attr, False)

    @classmethod
    def get_global_attr(cls) -> str:
        """Возвращает строковое представление глобального атрибута

        Returns:
            str: Строка обозначающая атрибут
        """
        return cls._global_meta_attr

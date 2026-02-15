from .core import CompilerDirective, nil_type
from .realm import Realm


@CompilerDirective.gmod_api("IsValid", [Realm.SHARED])
@CompilerDirective.typeguard_nil()
def IsValid(obj: object | nil_type) -> bool:
    """Возвращает значение, указывающее, является ли объект валидным или нет (например, сущности, панели, объекты пользовательских таблиц и многое другое).

    Проверяет, не равен ли объект `nil`, имеет ли он метод `IsValid` и возвращает ли этот метод значение `true`.

    Если у объекта нет метода `IsValid`, он вернет `false`."""
    ...

from .core import CompilerDirective, nil
from .realm import Realm


@CompilerDirective.gmod_api("IsValid", [Realm.SHARED, Realm.MENU])
@CompilerDirective.typeguard_nil()
def IsValid(obj: object | nil) -> bool:
    """TODO:"""
    ...

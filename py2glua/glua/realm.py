from enum import IntEnum


class Realm(IntEnum):
    """Область выполнения кода.

    `SERVER` - код выполняется только на сервере

    `CLIENT` - код выполняется только на клиенте

    `SHARED` - код выполняется и на сервере, и на клиенте
    """

    SERVER = 1
    CLIENT = 2
    SHARED = 3

    def __str__(self) -> str:
        return self.name.lower()

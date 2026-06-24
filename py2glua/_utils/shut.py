import sys
from typing import NoReturn

from .logger_init import logger


class Shutdown:
    @classmethod
    def normal(cls, reason: str | None = None) -> NoReturn:
        if reason:
            logger.info(reason)

        sys.exit(0)

    @classmethod
    def user_error(cls, reason: str) -> NoReturn:
        logger.error(reason)
        sys.exit(1)

    @classmethod
    def softwear_error(cls, reason: str) -> NoReturn:
        logger.error(reason, exc_info=True)
        sys.exit(2)

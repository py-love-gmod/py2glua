import logging

from .config import Config
from .logger_init import AlignedColorFormatter, log_step, logger
from .shut import Shutdown


def setup(debug: bool, verbouse: bool) -> None:
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(AlignedColorFormatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    Config.cli_setup(debug, verbouse)


__all__ = [
    "Config",
    "log_step",
    "logger",
    "Shutdown",
]

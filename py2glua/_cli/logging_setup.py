from __future__ import annotations

import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from .ansi import Fore, Style, init

init(autoreset=True)
logger = logging.getLogger("py2glua")


class AlignedColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
    }

    def __init__(self, fmt=None, datefmt=None, level_width=8):
        super().__init__(fmt, datefmt)
        self.level_width = level_width

    def format(self, record):
        if not isinstance(record.msg, str):
            record.msg = str(record.msg)

        levelname = record.levelname
        color = self.COLORS.get(levelname, "")
        padded_level = f"{color}{levelname:<{self.level_width}}{Style.RESET_ALL}"

        if "\n" in record.msg:
            lines = record.msg.splitlines()
            record.msg = ("\n" + " " * (self.level_width + 3)).join(lines)

        record.levelname = padded_level
        return super().format(record)


def setup_logging(debug: bool) -> logging.Logger:
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(AlignedColorFormatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


@contextmanager
def log_step(title: str, *, show_if_less_than: float = 0.002) -> Iterator[None]:
    logger.info(title)
    t0 = perf_counter()
    try:
        yield

    finally:
        dt = perf_counter() - t0
        if dt >= show_if_less_than:
            logger.info("Завершено (%.3fs)", dt)

        else:
            logger.info("Завершено")

        print()

import argparse
import logging
import sys
from importlib.metadata import PackageNotFoundError, version

from colorama import Fore, Style, init

init(autoreset=True)


class ColorFormater(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record) -> str:
        level_name = record.levelname
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{level_name}{Style.RESET_ALL}"
        return super().format(record)


def _safe_version() -> str:
    try:
        return version("py2glua")

    except PackageNotFoundError:
        return "0.0.0-dev"


def _build_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("py2glua")

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormater("[%(levelname)s]: %(message)s"))
    logger.addHandler(handler)

    return logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py2glua",
        description="Утилита для компиляции python -> glua",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Переключает логи на более расширенный вывод",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logger = _build_logger(args.debug)
    logger.debug("py2glua")
    logger.debug(f"Version: {_safe_version()}")

    sys.exit(0)


if __name__ == "__main__":
    main()

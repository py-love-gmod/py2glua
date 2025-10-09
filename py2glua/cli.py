import argparse
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from colorama import Fore, Style, init

from .compiler import Compiler

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
        return "dev"


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

    subparsers = parser.add_subparsers(dest="command", required=True)

    # команда build
    build_parser = subparsers.add_parser("build", help="Компилирует проект")
    build_parser.add_argument(
        "source",
        type=Path,
        help="Путь к директории с исходными .py файлами",
    )
    build_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Путь для сохранения результата (если не указан - печать в консоль)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logger = _build_logger(args.debug)
    logger.debug("[cli] py2glua")
    logger.debug(f"[cli] Version: {_safe_version()}")

    if args.command == "build":
        logger.info("[cli] Запуск компиляции...")
        compiler = Compiler(args.source, args.output)
        compiler.run()
        logger.info("[cli] Компиляция завершена")

    sys.exit(0)


if __name__ == "__main__":
    main()

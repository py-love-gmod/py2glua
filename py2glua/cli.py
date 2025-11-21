import argparse
import logging
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from colorama import Fore, Style, init

init(autoreset=True)


# region Logger and color
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


logger = logging.getLogger("py2glua")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
formatter = AlignedColorFormatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# endregion


def _verison() -> str:
    try:
        return version("py2glua")

    except PackageNotFoundError:
        return "0.0.0dev"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py2glua",
        description="Python -> GLua транслитор",
    )

    # region Main args
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Включить режим отладки",
    )
    # endregion

    # region Build cmd
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Собирает python код в glua код")
    b.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=None,
        dest="build_debug",
        help="Переключает режим билда с релизного на дебаг",
    )

    b.add_argument(
        "-c",
        "--clean_build",
        action="store_true",
        default=None,
        help="Очищать ли папку out перед сборкой",
    )

    b.add_argument(
        "-o",
        "--optimization",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Уровень оптимизации при трансляции",
    )

    b.add_argument(
        "src",
        type=Path,
        nargs="?",
        default=None,
        help="Исходная папка для исходного кода (по умолчанию: ./source)",
    )

    b.add_argument(
        "out",
        type=Path,
        nargs="?",
        default=None,
        help="Папка для собранного кода (по умолчанию: ./build)",
    )
    # endregion

    return parser


def _clean_build(src: Path | None) -> None:
    if src and src.exists():
        shutil.rmtree(src)


def _build(src: Path, out: Path, args) -> None:
    pass


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    logger.debug(f"Py2Glua\nVerison: {_verison()}")

    try:
        if args.cmd == "build":
            logger.debug(f"Start build\nSRC: {args.src}\nOUT: {args.out}")
            _build(args.src, args.out, args)

        sys.exit(0)

    except KeyError as err:
        logger.error(f"Error accessing key {err}", exc_info=True)
        logger.error("Exit code 2")
        sys.exit(2)

    except Exception as err:
        logger.error(str(err), exc_info=True)
        logger.error("Exit code 3")
        sys.exit(3)

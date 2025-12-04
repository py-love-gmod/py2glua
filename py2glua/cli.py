import argparse
import logging
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from colorama import Fore, Style, init

from ._lang.compile import CompileError, Compiler

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


def _version() -> str:
    try:
        return version("py2glua")

    except PackageNotFoundError:
        return "0.0.0dev"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py2glua",
        description="Python -> GLua компилятор",
    )

    # region Main args
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Включить режим отладки",
    )
    # endregion

    sub = parser.add_subparsers(dest="cmd", required=True)

    # region Build cmd
    b = sub.add_parser("build", help="Собирает python код в glua код")

    b.add_argument(
        "src",
        type=Path,
        nargs="?",
        default=None,
        help="Исходная папка для исходного кода (по умолчанию: ./source)",
    )

    b.add_argument(
        "-a",
        "--author",
        type=str,
        help="Автор проекта",
    )

    b.add_argument(
        "-p",
        "--project",
        type=str,
        help="Имя проекта",
    )

    b.add_argument(
        "out",
        type=Path,
        nargs="?",
        default=None,
        help="Папка для собранного кода (по умолчанию: ./build)",
    )
    # endregion

    # region Version cmd
    sub.add_parser("version", help="Показывает версию py2glua")
    # endregion

    return parser


def _clean_build(src: Path | None) -> None:
    if src and src.exists():
        logger.debug("Очистка build директории...")
        shutil.rmtree(src)


def _build(src: Path, out: Path, args) -> None:
    logger.info("Начало сборки...")

    if src is None:
        src = Path("./source")

    if out is None:
        out = Path("./build")

    src = src.resolve()
    out = out.resolve()

    project_name = args.project
    author_name = args.author

    if not src.exists():
        raise FileNotFoundError(f"Исходная папка не найдена: {src}")

    py_files = sorted(p for p in src.rglob("*.py"))
    if not py_files:
        logger.warning("В папке нет .py файлов - нечего собирать")
        return

    logger.info(f"Найдено файлов: {len(py_files)}")

    compiler = Compiler(
        project_root=src,
        config={
            "version": _version(),
            "method_renames": {"__init__": "new"},
            "name_space": f"{project_name}_{author_name}",
        },
        file_passes=[],
        project_passes=[],
    )

    project_ir = compiler.build(py_files)

    _clean_build(out)
    out.mkdir(parents=True, exist_ok=True)
    for ir in project_ir:
        print(ir)  # TODO make normal output

    logger.info("Сборка успешно завершена.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    logger.debug(f"Py2Glua\nVersion: {_version()}")

    try:
        if args.cmd == "build":
            logger.debug(f"Start build\nSRC: {args.src}\nOUT: {args.out}")
            _build(args.src, args.out, args)

        elif args.cmd == "version":
            print(_version())

        sys.exit(0)

    except CompileError as err:
        logger.error(err)
        logger.error("Exit code 1")
        sys.exit(1)

    except SyntaxError as err:
        logger.error(err)
        logger.error("Exit code 1")
        sys.exit(1)

    except KeyError as err:
        logger.error(f"Error accessing key {err}", exc_info=True)
        logger.error("Exit code 2")
        sys.exit(2)

    except Exception as err:
        logger.error(str(err), exc_info=True)
        logger.error("Exit code 3")
        sys.exit(3)

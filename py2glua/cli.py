import argparse
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ._cli.logging_setup import exit_with_code, logger, setup_logging
from ._lang.compiler import Compiler


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
        default=Path("./source"),
        help="Исходная папка для исходного кода (по умолчанию: ./source)",
    )

    b.add_argument(
        "-n",
        "--namespace",
        type=str,
        help="Неймспейс проекта",
    )

    b.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("./build"),
        help="Папка для результата (по умолчанию: ./build)",
    )

    # endregion

    # region Version cmd
    sub.add_parser("version", help="Показывает версию py2glua")
    # endregion

    return parser


def _clean_build(out: Path) -> None:
    if out.exists():
        if not out.is_dir():
            exit_with_code(2, f"Путь build не является директорией: {out}")

        logger.debug("Очистка build директории...")
        shutil.rmtree(out)


def _build(src: Path, out: Path) -> None:
    logger.info("Начало сборки...")

    src = src.resolve()
    out = out.resolve()

    project_ir = Compiler.build(project_root=src)

    _clean_build(out)
    out.mkdir(parents=True, exist_ok=True)

    for ir in project_ir:
        print(ir)

    logger.info("Сборка успешно завершена.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(args.debug)

    logger.debug(f"Py2Glua\nVersion: {_version()}")

    if args.cmd == "build":
        _build(args.src, args.out)
        exit_with_code(0)

    elif args.cmd == "version":
        print(_version())
        exit_with_code(0)

    else:
        exit_with_code(1, "Неизвестный параметр консоли")

    exit_with_code(3)

import argparse
import shutil
from pathlib import Path

from ._cli.logging_setup import exit_with_code, logger, setup_logging
from ._lang.compiler import Compiler
from ._lang.lua_emiter import LuaEmitter
from .config import Py2GluaConfig


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


def _build(src: Path, out: Path) -> None:
    logger.info("Начало сборки...")

    project_ir = Compiler.build()

    if out.exists():
        if not out.is_dir():
            exit_with_code(2, f"Путь build не является директорией: {out}")

        logger.debug("Очистка build директории...")
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    emitter = LuaEmitter()

    for ir in project_ir:
        if ir.path is None:
            exit_with_code(3, "Внутренняя ошибка: IR-файл без path")

        rel = ir.path.relative_to(src)  # type: ignore
        out_path = out / rel.with_suffix(".lua")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        result = emitter.emit_file(ir)

        out_path.write_text(result.code, encoding="utf-8")
        logger.info(f"Сгенерирован файл: {out_path}")

    logger.info("Сборка успешно завершена.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(args.debug)
    Py2GluaConfig.debug = args.debug
    Py2GluaConfig.source = args.src.resolve()
    Py2GluaConfig.output = args.out.resolve()

    logger.debug(f"""Py2Glua
Version: {Py2GluaConfig.version()}
Source : {Py2GluaConfig.source}
Output : {Py2GluaConfig.output}
Debug  : {Py2GluaConfig.debug}
""")

    match args.cmd:
        case "build":
            _build(Py2GluaConfig.source, Py2GluaConfig.output)
            exit_with_code(0)

        case "version":
            print(Py2GluaConfig.version())
            exit_with_code(0)

        case _:
            exit_with_code(1, "Неизвестный параметр консоли")

    exit_with_code(3)

import argparse
import shutil
from pathlib import Path

from ._cli import CompilerExit, log_step, logger, setup_logging
from ._lang.compiler import Compiler
from ._lang.lua_emiter import LuaEmitter
from .config import Py2GluaConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py2glua",
        description="Python -> GLua компилятор",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # region Build cmd
    b = sub.add_parser("build", help="Собирает python код в glua код")

    b.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Подробное логирование",
    )

    b.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug режим сборки",
    )

    b.add_argument(
        "src",
        type=Path,
        nargs="?",
        default=Path("./source"),
        help="Исходная папка для исходного кода (по умолчанию: ./source)",
    )

    b.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("./build"),
        help="Папка для результата (по умолчанию: ./build)",
    )

    b.add_argument(
        "-n",
        "--namespace",
        type=str,
        help="Неймспейс проекта",
    )

    # endregion

    # region Version cmd
    sub.add_parser("version", help="Показывает версию py2glua")
    # endregion

    return parser


def _build(src: Path, out: Path) -> None:
    project_ir = Compiler.build()

    emitter = LuaEmitter()

    with log_step("[5/5] Генерация GLua"):
        if out.exists():
            if not out.is_dir():
                CompilerExit.system_error(f"Путь build не является директорией: {out}")

            logger.debug("Очистка build директории...")
            shutil.rmtree(out)

        out.mkdir(parents=True, exist_ok=True)

        for ir in project_ir:
            if ir.path is None:
                CompilerExit.internal_error("Внутренняя ошибка: IR-файл без path")

            rel = ir.path.relative_to(src)  # type: ignore
            out_path = out / rel.with_suffix(".lua")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            result = emitter.emit_file(ir)

            out_path.write_text(result.code, encoding="utf-8")

    logger.info("Сборка завершена")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    Py2GluaConfig.verbose = args.verbose
    Py2GluaConfig.debug = args.debug

    Py2GluaConfig.source = args.src.resolve()
    Py2GluaConfig.output = args.out.resolve()

    logger.debug(f"""Py2Glua
Version: {Py2GluaConfig.version()}
Source : {Py2GluaConfig.source}
Output : {Py2GluaConfig.output}
Verbose: {Py2GluaConfig.verbose}
Debug  : {Py2GluaConfig.debug}
""")

    match args.cmd:
        case "build":
            _build(Py2GluaConfig.source, Py2GluaConfig.output)
            CompilerExit.generic(0)

        case "version":
            print(Py2GluaConfig.version())
            CompilerExit.generic(0)

        case _:
            CompilerExit.user_error(
                "Неизвестный параметр консоли",
                show_path=False,
                show_pos=False,
            )

    CompilerExit.internal_error("Недостижимо")

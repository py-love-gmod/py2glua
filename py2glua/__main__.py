import argparse
import re
import secrets
import string
from pathlib import Path

from ._cli import CompilerExit, logger, setup_logging
from ._config import Py2GluaConfig
from ._lang.compiler import Compiler

_LUA_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _gen_namespace() -> str:
    alphabet = string.ascii_lowercase
    suffix = "".join(secrets.choice(alphabet) for _ in range(8))
    return f"_p2g_{suffix}"


def _validate_namespace(ns: str) -> None:
    if _LUA_IDENT_RE.match(ns):
        return

    if not ns:
        reason = "namespace пустой"

    elif not re.match(r"^[A-Za-z_]", ns):
        reason = "namespace должен начинаться с буквы или '_'"

    elif not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ns):
        reason = (
            "namespace может содержать только буквы, цифры и '_' "
            "и не может содержать точки, дефисы или пробелы"
        )

    else:
        reason = "неизвестная ошибка формата"

    CompilerExit.user_error(
        f"Некорректный namespace: {ns}\nПричина: {reason}",
        show_path=False,
        show_pos=False,
    )


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
        help="Неймспейс проекта (Lua identifier)",
    )
    # endregion

    # region Version cmd
    sub.add_parser("version", help="Показывает версию py2glua")
    # endregion

    return parser


def _build() -> None:
    Compiler.build()
    logger.info("Сборка завершена")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    verbose = getattr(args, "verbose", False)
    setup_logging(verbose)

    match args.cmd:
        case "build":
            Py2GluaConfig.verbose = verbose
            Py2GluaConfig.debug = args.debug

            Py2GluaConfig.source = args.src.resolve()
            Py2GluaConfig.output = args.out.resolve()

            if args.namespace:
                _validate_namespace(args.namespace)
                Py2GluaConfig.namespace = args.namespace

            else:
                ns = _gen_namespace()
                Py2GluaConfig.namespace = ns
                logger.warning(
                    "Namespace не задан\n"
                    f"Namespace задан автоматически: {ns}\n"
                    "Установить свой можно через параметр -n | --namespace"
                )

            logger.debug(
                (
                    "Py2Glua\n"
                    f"Version  : {Py2GluaConfig.version()}\n"
                    f"Source   : {Py2GluaConfig.source}\n"
                    f"Output   : {Py2GluaConfig.output}\n"
                    f"Namespace: {Py2GluaConfig.namespace}\n"
                    f"Verbose  : {Py2GluaConfig.verbose}\n"
                    f"Debug    : {Py2GluaConfig.debug}"
                )
            )
            _build()
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


if __name__ == "__main__":
    main()

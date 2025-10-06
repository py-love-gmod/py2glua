import argparse
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from colorama import Fore, Style, init

from .compiler import Compiler
from .exceptions import CompileError
from .exceptions import NameError as GuardError


def _safe_version() -> str:
    try:
        return version("py2glua")

    except PackageNotFoundError:
        return "0.0.0-dev"


def _clean_build_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(
            f"[{Fore.CYAN}clean{Style.RESET_ALL}] removed old build directory: {path}"
        )


def _build_file(comp, py_file, out_path, fatal) -> None:
    try:
        comp.compile_file(py_file, out_path=out_path)
        print(f"[{Fore.GREEN}build{Style.RESET_ALL}] {py_file} -> {out_path}")

    except (CompileError, GuardError) as e:
        print(f"[{Fore.RED}error{Style.RESET_ALL}] {e}")
        if fatal:
            sys.exit(1)

    except Exception as e:
        print(
            f"[{Fore.MAGENTA}fatal{Style.RESET_ALL}] unexpected error: {e.__class__.__name__}: {e}"
        )
        if fatal:
            sys.exit(1)


def main() -> None:
    init(autoreset=True)
    parser = argparse.ArgumentParser(
        prog="py2glua",
        description="Python -> GLua транслитор",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Собирает python код в glua код")
    b.add_argument(
        "src",
        type=Path,
        nargs="?",
        default=Path("source"),
        help="Исходная папка для исходного кода (по умолчанию: ./source)",
    )
    b.add_argument(
        "out",
        type=Path,
        nargs="?",
        default=Path("build"),
        help="Папка для собранного кода (по умолчанию: ./build)",
    )
    b.add_argument(
        "--no-header",
        action="store_true",
        help="Не вставлять в файлы метаданные (по умолчанию: False)",
    )
    b.add_argument(
        "--no-clean",
        action="store_true",
        help="Не очищать папку сборки перед компиляцией (по умолчанию: False)",
    )
    b.add_argument(
        "--no-fatal",
        action="store_true",
        help="Пытается собрать все файла, а не падать после первой ошибки (по умолчанию: False)",
    )

    args = parser.parse_args()

    if args.cmd == "build":
        comp = Compiler(version=_safe_version(), add_header=not args.no_header)

        if not args.no_clean and args.out.exists():
            _clean_build_dir(args.out)

        args.out.mkdir(parents=True, exist_ok=True)

        if args.src.is_dir():
            for py_file in args.src.rglob("*.py"):
                rel = py_file.relative_to(args.src)
                lua_path = args.out / rel.with_suffix(".lua")
                lua_path.parent.mkdir(parents=True, exist_ok=True)
                _build_file(comp, py_file, lua_path, not args.no_fatal)

        else:
            out_path = args.out
            if out_path.is_dir():
                out_path = out_path / args.src.with_suffix(".lua").name
                _build_file(comp, args.src, out_path, not args.no_fatal)

    print(f"[{Fore.CYAN}done {Style.RESET_ALL}] Build done")
    sys.exit(0)

import argparse
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .convertor.compiler import Compiler


def _safe_version() -> str:
    try:
        return version("py2glua")

    except PackageNotFoundError:
        return "0.0.0-dev"


def _clean_build_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"[clean] removed old build directory: {path}")


def main() -> None:
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
                comp.compile_file(py_file, out_path=lua_path)
                print(f"[built] {py_file} -> {lua_path}")

        else:
            out_path = args.out
            if out_path.is_dir():
                out_path = out_path / args.src.with_suffix(".lua").name

            comp.compile_file(args.src, out_path=out_path)
            print(f"[built] {args.src} -> {out_path}")

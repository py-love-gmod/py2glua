import argparse
from importlib.metadata import version
from pathlib import Path

from .convertor.compiler import Compiler


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="py2glua", description="Python -> GLua transpiler"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Compile a Python file to Lua")
    b.add_argument("src", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--no-header", action="store_true")

    args = parser.parse_args()

    if args.cmd == "build":
        comp = Compiler(version=version("py2glua"), add_header=not args.no_header)
        comp.compile_file(args.src, out_path=args.out)
        print(f"Built {args.src.name} -> {args.out}")

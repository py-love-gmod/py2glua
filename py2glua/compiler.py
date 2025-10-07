from __future__ import annotations

import ast
from pathlib import Path
from typing import List

from .analyzer import NameGuard
from .convertor import (
    AssignEmitter,
    BinOpEmitter,
    FunctionEmitter,
    ReturnEmitter,
)


def sha1_text(s: str) -> str:
    import hashlib

    return hashlib.sha1(s.encode("utf-8")).hexdigest().upper()


def sha1_of_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest().upper()


def make_header(
    src_path: Path,
    version: str,
    repo_url: str,
    out_hash: str,
) -> str:
    from datetime import datetime

    src_hash = sha1_of_file(src_path)
    return (
        "-- ============================================\n"
        "--  py2glua build metadata\n"
        f"--  Version:       {version}\n"
        f"--  Repository:    {repo_url}\n"
        f"--  Source:        {src_path.as_posix()}\n"
        f"--  SHA1(src):     {src_hash}\n"
        f"--  SHA1(out):     {out_hash}\n"
        f"--  Build date:    {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        "-- ============================================\n\n"
    )


class Compiler:
    def __init__(
        self,
        *,
        version: str = "0.0.1",
        repo_url: str = "https://github.com/themanyfaceddemon/py2glua",
        add_header: bool = True,
    ) -> None:
        self.version = version
        self.repo_url = repo_url
        self.add_header = add_header

        self.func_emitter = FunctionEmitter()
        self.assign_emitter = AssignEmitter()
        self.return_emitter = ReturnEmitter()
        self.binop_emitter = BinOpEmitter()

    def compile_str(self, code: str, *, module_name: str = "<memory>") -> str:
        tree = ast.parse(code, filename=module_name, mode="exec")
        return self._emit_module(tree, filename=module_name)

    def compile_file(self, src_path: Path, *, out_path: Path | None = None) -> str:
        code = src_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(src_path), mode="exec")
        body = self._emit_module(tree, filename=str(src_path))

        if not self.add_header:
            return body

        out_hash = sha1_text(body)
        header = make_header(
            src_path=src_path,
            version=self.version,
            repo_url=self.repo_url,
            out_hash=out_hash,
        )
        final = header + body

        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(final, encoding="utf-8")

        return final

    def _emit_module(self, module: ast.Module, filename: str | None = None) -> str:
        lines: List[str] = []
        NameGuard().validate(module, filename=filename)

        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                lines.append(self.func_emitter.emit(node, indent=0))

            elif isinstance(node, ast.Assign):
                lines.append(self.assign_emitter.emit(node, indent=0))

            elif isinstance(node, ast.Expr):
                pass

            else:
                pass

        return "\n\n".join(lines).rstrip() + ("\n" if lines else "")

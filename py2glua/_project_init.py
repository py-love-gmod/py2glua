from __future__ import annotations

from pathlib import Path

from ._api_build import build_api_bundle
from ._cli import logger


def init_project(*, source: Path, output: Path, lang: str) -> None:
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    _ensure_file(source / "__init__.py", "")

    pkg_root = Path(__file__).resolve().parent
    data_root = pkg_root / "data"
    api_build_root = pkg_root / "build_api"

    out_api = build_api_bundle(
        data_root=data_root,
        build_root=api_build_root,
        lang=lang,
    )

    msg = (
        "Init завершён.\n"
        f"Source: {source}\n"
        f"Output: {output}\n"
        f"API module: {out_api}"
    )
    logger.info(msg)


def _ensure_file(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")

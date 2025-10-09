import ast
import logging
from pathlib import Path

from .analyzer import *  # noqa: F403
from .managers import AnalyzerManager

logger = logging.getLogger("py2glua")


class Compiler:
    def __init__(self, source: Path, output: Path | None = None):
        self.source = source
        self.output = output
        self.files: list[Path] = []
        self.results: dict[str, dict] = {}

    def _collect_files(self) -> None:
        if not self.source.exists():
            logger.warning(f"[compiler] путь {self.source} не существует")
            return

        if not self.source.is_dir():
            logger.warning(f"[compiler] путь {self.source} не является папкой")
            return

        self.files = [p for p in self.source.rglob("*.py")]
        logger.info(f"[compiler] найдено {len(self.files)} файлов")

    def _analyze_file(self, path: Path) -> dict:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))

        ctx = {
            "aliases": {},
        }

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                unit = AnalyzerManager.get(node)
                if unit:
                    unit.process(node, ctx)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue

            unit = AnalyzerManager.get(node)
            if unit:
                unit.process(node, ctx)

            else:
                logger.warning(f"[compiler] неизвестный ast тип {type(node)}")

        return {}

    def run(self) -> None:
        self._collect_files()

        for file in self.files:
            try:
                info = self._analyze_file(file)
                self.results[str(file)] = info

            except SyntaxError as e:
                logger.error(f"[compiler] ошибка парсинга {file}: {e}")

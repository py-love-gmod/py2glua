import logging
import tomllib
from pathlib import Path
from typing import Any

from .optimyzer import OptimizationLevel

logger = logging.getLogger("py2glua")


class Config:
    _data: dict[str, Any] = {}

    @classmethod
    def _warn(cls, msg: str, source: Path):
        logger.warning(
            f"[config] {msg}\nPath: {source}\nInfo: Будут использоваться базовые настройки"
        )

    @classmethod
    def get(cls, path: str, default: Any = None) -> Any:
        value = cls._data
        for part in path.split("."):
            if not isinstance(value, dict):
                return default

            value = value.get(part, default)

        return value

    @classmethod
    def reload(cls, source: Path) -> None:
        cls._data.clear()
        cls.load(source)

    @classmethod
    def load(cls, source: Path) -> None:
        if not source.exists():
            cls._warn("Конфиг не найден", source)
            return

        with source.open("rb") as f:
            data = tomllib.load(f)

        if not data.get("py2glua", None):
            cls._warn("Секция py2glua не найдена", source)
            return

        ver = data["py2glua"].get("version", None)
        if ver is None:
            cls._warn("В секции py2glua не найдена версия конфига", source)
            return

        func = getattr(cls, f"_{ver}", None)
        if func is None or not callable(func):
            cls._warn(f"Версия {ver} не поддерживается", source)
            return

        func(data)

    @classmethod
    def _v1(cls, data: dict[str, Any]) -> None:
        proj = data.get("project", {})
        comp = data.get("compiler", {})

        cls._data["project"] = {
            "name": proj.get("name"),
            "author": proj.get("author"),
        }

        cls._data["compiler"] = {
            "optimization": OptimizationLevel(int(comp.get("optimization", 2))),
            "input": Path(comp.get("input", "source")),
            "output": Path(comp.get("output", "build")),
        }

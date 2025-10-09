import logging
from pathlib import Path

import pytest

from py2glua.config import Config
from py2glua.optimyzer import OptimizationLevel


@pytest.fixture(autouse=True)
def clear_config():
    Config._data.clear()
    yield
    Config._data.clear()


def make_toml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "test_config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_valid_v1(tmp_path):
    toml_data = """
[py2glua]
version = "v1"

[project]
name = "TestProject"
author = "Cain"

[compiler]
optimization = 3
input = "src"
output = "out"
"""
    path = make_toml(tmp_path, toml_data)
    Config.load(path)

    assert Config.get("project.name") == "TestProject"
    assert Config.get("project.author") == "Cain"

    comp = Config._data["compiler"]
    assert isinstance(comp["optimization"], OptimizationLevel)
    assert comp["optimization"].value == 3
    assert comp["input"] == Path("src")
    assert comp["output"] == Path("out")


def test_load_missing_file(tmp_path, caplog):
    fake = tmp_path / "missing.toml"
    with caplog.at_level(logging.WARNING):
        Config.load(fake)

    assert "Конфиг не найден" in caplog.text
    assert Config._data == {}


def test_load_missing_py2glua_section(tmp_path, caplog):
    toml_data = """
[project]
name = "x"
"""
    path = make_toml(tmp_path, toml_data)
    with caplog.at_level(logging.WARNING):
        Config.load(path)

    assert "Секция py2glua не найдена" in caplog.text
    assert Config._data == {}


def test_load_unsupported_version(tmp_path, caplog):
    toml_data = """
[py2glua]
version = "v999"
"""
    path = make_toml(tmp_path, toml_data)
    with caplog.at_level(logging.WARNING):
        Config.load(path)

    assert "не поддерживается" in caplog.text
    assert Config._data == {}


def test_get_nested_default():
    Config._data = {"a": {"b": {"c": 5}}}
    assert Config.get("a.b.c") == 5
    assert Config.get("a.b.x", 42) == 42
    assert Config.get("a.b.c.d", "none") == "none"
    assert Config.get("missing", "ok") == "ok"


def test_reload(tmp_path):
    toml_data = """
[py2glua]
version = "v1"

[project]
name = "ReloadTest"
"""
    path = make_toml(tmp_path, toml_data)
    Config.load(path)
    assert "ReloadTest" in Config.get("project.name")
    Config._data["project"]["name"] = "Changed"
    Config.reload(path)
    assert Config.get("project.name") == "ReloadTest"

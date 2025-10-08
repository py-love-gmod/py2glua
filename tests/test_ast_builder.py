import ast
from pathlib import Path

from py2glua.ast_builder import ASTBuilder

TESTS = Path(__file__).parent


def make_temp_file(tmp_path, text: str, name="test.py") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8-sig")
    return p


def test_valid_python(tmp_path):
    file = make_temp_file(tmp_path, "x = 42\nprint(x)\n")
    result = ASTBuilder.load_from_file(file)
    assert isinstance(result, ast.Module)
    assert len(result.body) == 2


def test_nonexistent_file(tmp_path, caplog):
    fake = tmp_path / "missing.py"
    result = ASTBuilder.load_from_file(fake)
    assert result is None
    assert any("не существует" in rec.message for rec in caplog.records)


def test_non_python_file(tmp_path, caplog):
    file = make_temp_file(tmp_path, "print('x')", "text.txt")
    result = ASTBuilder.load_from_file(file)
    assert result is None
    assert any("не является" in rec.message for rec in caplog.records)


def test_syntax_error(tmp_path, caplog):
    file = make_temp_file(tmp_path, "def f(:\n    pass\n")
    result = ASTBuilder.load_from_file(file)
    assert result is None
    assert any("Синтаксическая ошибка" in rec.message for rec in caplog.records)


def test_broken_encoding(tmp_path, caplog):
    file = tmp_path / "enc.py"
    file.write_bytes(b"\xff\xfe\x00\x00")  # мусор
    result = ASTBuilder.load_from_file(file)
    assert result is None
    assert any("Ошибка декодирования" in rec.message for rec in caplog.records)

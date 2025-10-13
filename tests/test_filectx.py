import ast
import logging
import textwrap
from pathlib import Path

from py2glua.file_meta import Priority
from py2glua.files.filectx import FileCTX, _MagicIntResult
from py2glua.runtime import Realm


def test_load_ast_success(tmp_path: Path):
    src = tmp_path / "a.py"
    src.write_text("x = 1")
    ctx = FileCTX()
    assert ctx._load_ast(src)
    assert isinstance(ctx.file_ast, ast.Module)
    assert ctx.file_path == src


def test_load_ast_nonexistent(tmp_path: Path, caplog):
    bad = tmp_path / "nope.py"
    ctx = FileCTX()
    with caplog.at_level(logging.ERROR):
        ok = ctx._load_ast(bad)

    assert not ok
    assert "file dont exists" in caplog.text


def test_load_ast_wrong_ext(tmp_path: Path, caplog):
    bad = tmp_path / "x.txt"
    bad.write_text("meh")
    ctx = FileCTX()
    with caplog.at_level(logging.ERROR):
        ok = ctx._load_ast(bad)

    assert not ok
    assert "not a .py file" in caplog.text


def test_resolve_imports_simple():
    ctx = FileCTX()
    src = textwrap.dedent("""
        import a.b as c
        from x.y import z as w
        c.foo()
        w()
    """)

    ctx.file_ast = ast.parse(src)
    ctx._resolve_imports()
    code = ast.unparse(ctx.file_ast)
    assert "a.b.foo" in code
    assert "x.y.z" in code


def test_remove_main_block():
    ctx = FileCTX()
    src = textwrap.dedent("""
        if __name__ == "__main__":
            print("hello")
        print("bye")
    """)
    ctx.file_ast = ast.parse(src)
    ctx._remove_enter_point()
    code = ast.unparse(ctx.file_ast)
    assert "hello" not in code
    assert "bye" in code


def test_extract_magic_int_found():
    ctx = FileCTX()
    ctx.file_ast = ast.parse("__realm__ = 2")
    res = ctx._extract_magic_int("__realm__")
    assert isinstance(res, _MagicIntResult)
    assert res.value == 2


def test_extract_magic_int_not_found():
    ctx = FileCTX()
    ctx.file_ast = ast.parse("x = 1")
    assert ctx._extract_magic_int("__realm__") is None


def test_get_realm_valid(monkeypatch):
    ctx = FileCTX()
    ctx.file_ast = ast.parse("__realm__ = 2")
    assert ctx._get_realm().value == 2


def test_get_realm_invalid(caplog):
    ctx = FileCTX()
    ctx.file_path = Path("dummy.py")
    ctx.file_ast = ast.parse("__realm__ = 99")
    with caplog.at_level(logging.ERROR):
        val = ctx._get_realm()

    assert val.value == Realm.SERVER.value
    assert "invalid __realm__" in caplog.text


def test_get_priority_present():
    ctx = FileCTX()
    ctx.file_ast = ast.parse("__priority__ = 10")
    assert ctx._get_priority() == 10


def test_get_priority_missing():
    ctx = FileCTX()
    ctx.file_ast = ast.parse("x = 1")
    assert ctx._get_priority() == Priority.MEDIUM.value


def test_cleanup_magic_ints_removes():
    ctx = FileCTX()
    ctx.file_ast = ast.parse("__realm__ = 1\n__priority__ = 2\nx = 3")
    ctx._cleanup_magic_ints(("__realm__", "__priority__"))
    code = ast.unparse(ctx.file_ast)
    assert "__realm__" not in code
    assert "__priority__" not in code
    assert "x = 3" in code

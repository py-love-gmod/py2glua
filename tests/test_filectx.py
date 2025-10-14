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
    ctx.file_path = Path("dummy.py")
    ctx._resolve_imports()
    code = ast.unparse(ctx.file_ast)
    assert "a.b.foo" in code
    assert "x.y.z" in code


def test_resolve_imports_relative(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sub.py").write_text("def foo(): pass")
    (tmp_path / "other.py").write_text("def bar(): pass")

    src = textwrap.dedent("""
        from .sub import foo
        from ..other import bar
        foo()
        bar()
    """)

    ctx = FileCTX()
    ctx.file_ast = ast.parse(src)
    ctx.file_path = pkg / "__init__.py"
    ctx._resolve_imports()

    code = ast.unparse(ctx.file_ast)

    assert "mypkg.sub.foo" in code
    assert "other.bar" in code


def test_build_file_imports(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sub.py").write_text("def foo(): pass")
    (tmp_path / "other.py").write_text("def bar(): pass")

    src = textwrap.dedent("""
        import sys
        import logging
        import mypkg.sub as sub
        import other
        sub.foo()
        other.bar()
    """)

    ctx = FileCTX()
    ctx.file_ast = ast.parse(src)
    ctx.file_path = pkg / "__init__.py"

    ctx._resolve_imports()
    ctx._build_file_imports(source_path=tmp_path)

    kinds = {imp.split("|")[0] for imp in ctx.file_imports}
    names = {imp.split("|")[1] for imp in ctx.file_imports}

    assert "std" in kinds
    assert "loc" in kinds
    assert "sys" in names
    assert "logging" in names
    assert "mypkg" in names
    assert "other" in names


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


def test_get_realm_shared_enum(tmp_path: Path):
    src_file = tmp_path / "realm_enum_shared.py"
    src_file.write_text(
        textwrap.dedent("""
        from py2glua.runtime import Realm
        __realm__ = Realm.SHARED
        """)
    )

    ctx = FileCTX()
    ctx.build(src_file, tmp_path)

    assert ctx.file_realm == Realm.SHARED
    code = ast.unparse(ctx.file_ast)
    assert "__realm__" not in code


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
    for name in ["__realm__", "__priority__"]:
        ctx._cleanup_magic_int(name)

    code = ast.unparse(ctx.file_ast)
    assert "__realm__" not in code
    assert "__priority__" not in code
    assert "x = 3" in code


def test_collect_globals_functions_and_classes(tmp_path: Path):
    src_file = tmp_path / "a.py"
    src_file.write_text(
        textwrap.dedent("""
        from py2glua import Global

        @Global.mark()
        def my_func(): pass

        @Global.mark()
        class MyClass: pass

        x = Global.var(1)
        y = Global.var(2)
    """)
    )

    ctx = FileCTX()
    ctx.build(src_file, tmp_path)

    globals_set = ctx.file_globals
    assert "func|my_func" in globals_set
    assert "class|MyClass" in globals_set
    assert "var|x" in globals_set
    assert "var|y" in globals_set
    assert len(globals_set) == 4


def test_collect_globals_after_import_normalization(tmp_path: Path):
    src_file = tmp_path / "a.py"
    src_file.write_text(
        textwrap.dedent("""
        import py2glua as pg

        @pg.Global.mark()
        def f(): pass

        @pg.Global.mark()
        class C: pass

        a = pg.Global.var(1)
    """)
    )

    ctx = FileCTX()
    ctx.build(src_file, tmp_path)

    globals_set = ctx.file_globals
    assert "func|f" in globals_set
    assert "class|C" in globals_set
    assert "var|a" in globals_set


def test_collect_globals_var_with_argument(tmp_path: Path, caplog):
    src_file = tmp_path / "globals_arg.py"
    src_file.write_text(
        textwrap.dedent("""
        from py2glua import Global

        x = Global.var("my_var")
        y = Global.var()
        z = Global.var("a", "b")
    """)
    )

    ctx = FileCTX()
    with caplog.at_level(logging.WARNING):
        ctx.build(src_file, tmp_path)

    globals_set = ctx.file_globals
    assert "var|x" in globals_set
    assert "var|y" not in globals_set
    assert "var|z" not in globals_set


def test_filectx_full_integration(tmp_path: Path):
    src_file = tmp_path / "full.py"
    src_file.write_text(
        textwrap.dedent("""
        from py2glua.runtime import Realm
        from py2glua import Global

        __realm__ = Realm.CLIENT
        __priority__ = 5

        @Global.mark()
        def my_func(): pass

        @Global.mark()
        class MyClass: pass

        a = Global.var(1)
        b = Global.var(2)
    """)
    )

    ctx = FileCTX()
    ctx.build(src_file, tmp_path)

    assert ctx.file_realm == Realm.CLIENT
    assert ctx.file_priority == 5

    globals_set = ctx.file_globals
    assert "func|my_func" in globals_set
    assert "class|MyClass" in globals_set
    assert "var|a" in globals_set
    assert "var|b" in globals_set
    assert len(globals_set) == 4

    code = ast.unparse(ctx.file_ast)
    assert "__realm__" not in code
    assert "__priority__" not in code

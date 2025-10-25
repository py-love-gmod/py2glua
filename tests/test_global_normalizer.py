import ast

from py2glua.config.py2glua_toml import Py2GluaConfig
from py2glua.ir.builder import IRBuilder
from py2glua.ir.ir_base import File, FunctionDef, VarStore
from py2glua.normalizer.global_normalizer import GlobalNormalizer
from py2glua.normalizer.import_normalizer import ImportNormalizer


def build_ir(src: str) -> File:
    tree = ast.parse(src)
    ir = IRBuilder.build_ir(tree)
    assert isinstance(ir, File)
    return ir


def test_global_var_simple():
    Py2GluaConfig.default()

    src = """from py2glua import Global
x = Global.var(1)
"""
    ir = build_ir(src)
    ImportNormalizer.run(ir)
    GlobalNormalizer.run(ir)

    assert "globals" in ir.meta
    globals_tbl = ir.meta["globals"]
    assert "x" in globals_tbl

    meta = globals_tbl["x"]
    assert meta["name"] == "x"
    assert meta["type"] == "var"
    assert meta["external"] is False
    assert isinstance(ir.body[1], VarStore)


def test_global_var_external_true():
    Py2GluaConfig.default()

    src = """from py2glua import Global
y = Global.var(1, external=True)
"""
    ir = build_ir(src)
    ImportNormalizer.run(ir)
    GlobalNormalizer.run(ir)

    globals_tbl = ir.meta["globals"]
    meta = globals_tbl["y"]

    assert meta["name"] == "y"
    assert meta["type"] == "var"
    assert meta["external"] is True


def test_function_mark_is_detected():
    Py2GluaConfig.default()

    src = """from py2glua import Global

@Global.mark()
def t1(): pass
"""
    ir = build_ir(src)
    ImportNormalizer.run(ir)
    GlobalNormalizer.run(ir)

    globals_tbl = ir.meta["globals"]
    assert "t1" in globals_tbl
    meta = globals_tbl["t1"]

    assert meta["type"] == "function"
    assert meta["external"] is False
    node = [n for n in ir.body if isinstance(n, FunctionDef)][0]
    assert node.lineno == meta["lineno"]


def test_function_mark_external_true():
    Py2GluaConfig.default()

    src = """from py2glua import Global

@Global.mark(external=True)
def t2(): pass
"""
    ir = build_ir(src)
    ImportNormalizer.run(ir)
    GlobalNormalizer.run(ir)

    globals_tbl = ir.meta["globals"]
    meta = globals_tbl["t2"]

    assert meta["type"] == "function"
    assert meta["external"] is True


def test_class_mark_and_function_mix():
    Py2GluaConfig.default()

    src = """from py2glua import Global

@Global.mark()
class A: pass

@Global.mark(external=True)
def f(): pass
"""
    ir = build_ir(src)
    ImportNormalizer.run(ir)
    GlobalNormalizer.run(ir)

    globals_tbl = ir.meta["globals"]

    assert set(globals_tbl.keys()) == {"A", "f"}

    class_meta = globals_tbl["A"]
    func_meta = globals_tbl["f"]

    assert class_meta["type"] == "class"
    assert class_meta["external"] is False
    assert func_meta["type"] == "function"
    assert func_meta["external"] is True


def test_no_global_entries_if_not_marked():
    Py2GluaConfig.default()

    src = """def f(): pass
x = 123
"""
    ir = build_ir(src)
    GlobalNormalizer.run(ir)

    assert "globals" not in ir.meta


def test_global_normalizer_absent_when_no_marks():
    Py2GluaConfig.default()
    ir = build_ir("def f():\n    pass\nx=1\n")
    GlobalNormalizer.run(ir)
    assert "globals" not in ir.meta

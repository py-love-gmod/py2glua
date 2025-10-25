import ast
from pathlib import Path

from py2glua.config.py2glua_toml import Py2GluaConfig
from py2glua.ir.builder import IRBuilder
from py2glua.ir.ir_base import (
    Attribute,
    Call,
    File,
    FunctionDef,
    Import,
    ImportType,
    VarLoad,
    VarStore,
)
from py2glua.normalizer.import_normalizer import ImportNormalizer


def build_ir(src: str) -> File:
    tree = ast.parse(src)
    ir = IRBuilder.build_ir(tree)
    assert isinstance(ir, File)
    return ir


def test_from_import_is_normalized_and_usage_replaced():
    Py2GluaConfig.default()
    ir = build_ir("from os import path\nvalue = path.join('a')\n")
    ImportNormalizer.run(ir)

    imp = ir.body[0]
    assert isinstance(imp, Import)
    assert imp.module is None
    assert imp.names == ["os.path"]
    assert imp.aliases == [None]
    assert imp.import_type is ImportType.PYTHON_STD

    store = ir.body[1]
    assert isinstance(store, VarStore)
    call = store.value
    assert isinstance(call, Call)
    func = call.func
    assert isinstance(func, Attribute) and func.attr == "join"
    path_attr = func.value
    assert isinstance(path_attr, Attribute) and path_attr.attr == "path"
    root = path_attr.value
    assert isinstance(root, VarLoad) and root.name == "os"

    assert "import_map" not in ir.meta


def test_from_import_alias_is_expanded():
    Py2GluaConfig.default()
    ir = build_ir("from collections import deque as dq\nqueue = dq([1])\n")
    ImportNormalizer.run(ir)

    imp = ir.body[0]
    assert isinstance(imp, Import)
    assert imp.module is None
    assert imp.names == ["collections.deque"]
    assert imp.aliases == [None]
    assert imp.import_type is ImportType.PYTHON_STD

    store = ir.body[1]
    assert isinstance(store, VarStore)
    call = store.value
    assert isinstance(call, Call)
    assert isinstance(call.func, Attribute)
    assert call.func.attr == "deque"
    base = call.func.value
    assert isinstance(base, VarLoad) and base.name == "collections"


def test_plain_import_alias_is_resolved_to_module_name():
    Py2GluaConfig.default()
    ir = build_ir("import math as m\nvalue = m.sqrt(9)\n")
    ImportNormalizer.run(ir)

    imp = ir.body[0]
    assert isinstance(imp, Import)
    assert imp.module is None
    assert imp.names == ["math"]
    assert imp.aliases == ["m"]
    assert imp.import_type is ImportType.PYTHON_STD

    store = ir.body[1]
    assert isinstance(store, VarStore)
    call = store.value
    assert isinstance(call, Call)
    assert isinstance(call.func, Attribute)
    assert call.func.attr == "sqrt"
    base = call.func.value
    assert isinstance(base, VarLoad) and base.name == "math"


def test_internal_import_marked_and_usage_rewritten():
    Py2GluaConfig.default()
    ir = build_ir("from py2glua.utils import helper\nresult = helper()\n")
    ImportNormalizer.run(ir)

    imp = ir.body[0]
    assert isinstance(imp, Import)
    assert imp.module is None
    assert imp.names == ["py2glua.utils.helper"]
    assert imp.aliases == [None]
    assert imp.import_type is ImportType.INTERNAL

    store = ir.body[1]
    assert isinstance(store, VarStore)
    call = store.value
    assert isinstance(call, Call)
    assert isinstance(call.func, Attribute)
    assert call.func.attr == "helper"
    utils_attr = call.func.value
    assert isinstance(utils_attr, Attribute) and utils_attr.attr == "utils"
    root = utils_attr.value
    assert isinstance(root, VarLoad) and root.name == "py2glua"


def test_local_import_marked_local(tmp_path: Path):
    Py2GluaConfig.default()
    Py2GluaConfig.data["build"]["source"] = tmp_path
    (tmp_path / "localmod.py").write_text("", "utf-8")
    ir = build_ir("import localmod\n")
    ImportNormalizer.run(ir)
    imp = ir.body[0]
    assert isinstance(imp, Import)
    assert imp.import_type is ImportType.LOCAL


def test_unknown_import_marked_external():
    Py2GluaConfig.default()
    ir = build_ir("import thirdparty_pkg\n")
    ImportNormalizer.run(ir)
    imp = ir.body[0]
    assert isinstance(imp, Import)
    assert imp.import_type is ImportType.EXTERNAL


def test_global_mark_decorators_are_qualified():
    Py2GluaConfig.default()
    src = """from py2glua import Global
@Global.mark()
def t1(): pass
@Global.mark(external=True)
def t2(): pass
"""
    ir = build_ir(src)
    ImportNormalizer.run(ir)

    fn1 = ir.body[1]
    fn2 = ir.body[2]
    assert isinstance(fn1, FunctionDef) and isinstance(fn2, FunctionDef)
    assert fn1.decorators == ["py2glua.Global.mark()"]
    assert fn2.decorators == ["py2glua.Global.mark(external=True)"]

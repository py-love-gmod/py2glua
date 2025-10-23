import ast
from pathlib import Path

from py2glua.config.py2glua_toml import Py2GluaConfig
from py2glua.ir.builder import IRBuilder
from py2glua.ir.ir_base import (
    Attribute,
    Call,
    File,
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

    import_node = ir.body[0]
    assert isinstance(import_node, Import)
    assert import_node.module is None
    assert import_node.names == ["os.path"]
    assert import_node.aliases == [None]
    assert import_node.import_type is ImportType.PYTHON_STD

    store = ir.body[1]
    assert isinstance(store, VarStore)
    call = store.value
    assert isinstance(call, Call)
    join_attr = call.func
    assert isinstance(join_attr, Attribute)
    assert join_attr.attr == "join"
    path_attr = join_attr.value
    assert isinstance(path_attr, Attribute)
    assert path_attr.attr == "path"
    root = path_attr.value
    assert isinstance(root, VarLoad)
    assert root.name == "os"

    assert "import_map" not in ir.meta


def test_from_import_alias_is_expanded():
    Py2GluaConfig.default()

    ir = build_ir("from collections import deque as dq\nqueue = dq([1])\n")

    ImportNormalizer.run(ir)

    import_node = ir.body[0]
    assert isinstance(import_node, Import)
    assert import_node.module is None
    assert import_node.names == ["collections.deque"]
    assert import_node.aliases == [None]
    assert import_node.import_type is ImportType.PYTHON_STD

    store = ir.body[1]
    assert isinstance(store, VarStore)
    call = store.value
    assert isinstance(call, Call)
    func = call.func
    assert isinstance(func, Attribute)
    assert func.attr == "deque"
    base = func.value
    assert isinstance(base, VarLoad)
    assert base.name == "collections"


def test_plain_import_alias_is_resolved_to_module_name():
    Py2GluaConfig.default()

    ir = build_ir("import math as m\nvalue = m.sqrt(9)\n")

    ImportNormalizer.run(ir)

    import_node = ir.body[0]
    assert isinstance(import_node, Import)
    assert import_node.module is None
    assert import_node.names == ["math"]
    assert import_node.aliases == ["m"]
    assert import_node.import_type is ImportType.PYTHON_STD

    store = ir.body[1]
    assert isinstance(store, VarStore)
    call = store.value
    assert isinstance(call, Call)
    func = call.func
    assert isinstance(func, Attribute)
    assert func.attr == "sqrt"
    base = func.value
    assert isinstance(base, VarLoad)
    assert base.name == "math"


def test_internal_import_marked_and_usage_rewritten():
    Py2GluaConfig.default()

    ir = build_ir("from py2glua.utils import helper\nresult = helper()\n")

    ImportNormalizer.run(ir)

    import_node = ir.body[0]
    assert isinstance(import_node, Import)
    assert import_node.module is None
    assert import_node.names == ["py2glua.utils.helper"]
    assert import_node.aliases == [None]
    assert import_node.import_type is ImportType.INTERNAL

    store = ir.body[1]
    assert isinstance(store, VarStore)
    call = store.value
    assert isinstance(call, Call)
    func = call.func
    assert isinstance(func, Attribute)
    assert func.attr == "helper"
    utils_attr = func.value
    assert isinstance(utils_attr, Attribute)
    assert utils_attr.attr == "utils"
    root = utils_attr.value
    assert isinstance(root, VarLoad)
    assert root.name == "py2glua"


def test_local_import_marked_local(tmp_path: Path):
    Py2GluaConfig.default()
    Py2GluaConfig.data["build"]["source"] = tmp_path

    (tmp_path / "localmod.py").write_text("", "utf-8")

    ir = build_ir("import localmod\n")

    ImportNormalizer.run(ir)

    import_node = ir.body[0]
    assert isinstance(import_node, Import)
    assert import_node.import_type is ImportType.LOCAL


def test_unknown_import_marked_external():
    Py2GluaConfig.default()

    ir = build_ir("import thirdparty_pkg\n")

    ImportNormalizer.run(ir)

    import_node = ir.body[0]
    assert isinstance(import_node, Import)
    assert import_node.import_type is ImportType.EXTERNAL

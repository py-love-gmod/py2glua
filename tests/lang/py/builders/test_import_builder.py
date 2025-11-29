from pathlib import Path

import pytest

from py2glua._lang.py.ir_builder import PyIRBuilder
from py2glua._lang.py.ir_dataclass import PyIRImportType
from py2glua.config import Py2GluaConfig

Py2GluaConfig.source = Path("/root/project").resolve()


def test_simple_import():
    src = "import foo"
    ir = PyIRBuilder.build_file(src, Path("/root/project/test.py"))

    assert len(ir.body) == 1
    node = ir.body[0]

    assert node.modules == ["foo"]  # type: ignore
    assert node.i_type in (  # type: ignore
        PyIRImportType.LOCAL,
        PyIRImportType.STD_LIB,
        PyIRImportType.UNKNOWN,
    )


def test_multiple_imports():
    src = "import foo, bar"
    ir = PyIRBuilder.build_file(src, Path("/root/project/test.py"))

    assert len(ir.body) == 2
    assert ir.body[0].modules == ["foo"]  # type: ignore
    assert ir.body[1].modules == ["bar"]  # type: ignore


def test_import_as():
    src = "import foo as f"
    ir_file = PyIRBuilder.build_file(src, Path("/root/project/test.py"))

    ctx = ir_file.context.meta
    assert ctx["aliases"]["f"] == "foo"
    assert ir_file.body[0].modules == ["foo"]  # type: ignore


def test_from_import():
    src = "from pkg.sub import mod"
    ir_file = PyIRBuilder.build_file(src, Path("/root/project/test.py"))

    assert len(ir_file.body) == 1
    assert ir_file.body[0].modules == ["pkg.sub.mod"]  # type: ignore


def test_relative_import_single_dot():
    src = "from . import util"
    ir_file = PyIRBuilder.build_file(src, Path("/root/project/pkg/module.py"))

    assert ir_file.body[0].modules == ["pkg.util"]  # type: ignore


def test_relative_import_parent():
    src = "from ..utils import helper"
    ir_file = PyIRBuilder.build_file(src, Path("/root/project/pkg/sub/module.py"))

    assert ir_file.body[0].modules == ["pkg.utils.helper"]  # type: ignore


def test_forbid_relative_import_import_form():
    src = "import .foo"
    with pytest.raises(SyntaxError):
        PyIRBuilder.build_file(src, Path("/root/project/pkg/sub/module.py"))


def test_forbid_wildcard():
    src = "from pkg import *"
    with pytest.raises(SyntaxError):
        PyIRBuilder.build_file(src, Path("/root/project/a.py"))

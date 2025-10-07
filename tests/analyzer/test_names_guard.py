import ast

import pytest

from py2glua.analyzer.names_guard import NameGuard
from py2glua.exceptions import NameError


def test_rejects_lua_keywords():
    src = "def func():\n    local = 5\n    return local\n"
    tree = ast.parse(src)
    guard = NameGuard()
    with pytest.raises(NameError) as e:
        guard.validate(tree)

    msg = str(e.value)
    assert "local" in msg
    assert "Line 2" in msg


def test_rejects_system_globals():
    src = "def func():\n    table = {}\n    return table\n"
    tree = ast.parse(src)
    guard = NameGuard()
    with pytest.raises(NameError):
        guard.validate(tree)

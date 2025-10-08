import ast

from py2glua.convertor import ConvertorManager
from py2glua.convertor.base_convertor import BaseConvertor


def test_convertor_manager_registratiron():
    @ConvertorManager.register(ast.Assign)
    class TestConvertor(BaseConvertor): ...

    assert isinstance(ConvertorManager.get(ast.parse("x = 1").body[0]), BaseConvertor)

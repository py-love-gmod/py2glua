import pytest

from py2glua.glua import Global


def test_callable_decorator_marks_function():
    @Global.callable
    def sample_func(): ...

    assert getattr(sample_func, Global.get_global_attr(), False) is True
    assert Global.is_global(sample_func) is True


def test_callable_decorator_marks_class():
    @Global.callable
    class SampleClass: ...

    assert getattr(SampleClass, Global.get_global_attr(), False) is True
    assert Global.is_global(SampleClass) is True


def test_callable_does_not_double_set_attr():
    @Global.callable
    def func1(): ...

    @Global.callable
    def func2(): ...

    assert Global.is_global(func1)
    assert Global.is_global(func2)


def test_callable_double_set_attr():
    @Global.callable
    def func1(): ...

    Global.callable(func1)

    assert Global.is_global(func1)


def test_var_marks_object_with_attr():
    class Dummy: ...

    d = Dummy()
    Global.var(d)

    assert getattr(d, Global.get_global_attr(), False) is True
    assert Global.is_global(d) is True


def test_var_on_primitive_does_not_raise():
    val = Global.var(123)
    assert val == 123
    assert Global.is_global(val) is False


def test_is_global_false_by_default():
    class Dummy: ...

    d = Dummy()
    assert Global.is_global(d) is False


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        123,
        "text",
        [1, 2, 3],
        {1: 1, 2: 2, 3: 3},
    ],
)
def test_var_on_all_basic_types(value):
    assert Global.var(value) == value
    assert Global.is_global(value) is False

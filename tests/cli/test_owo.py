import pytest

from source.py2glua import cli


def test_main_owofy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "OwO"

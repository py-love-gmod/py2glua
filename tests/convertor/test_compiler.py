from pathlib import Path

from source.py2glua.convertor.compiler import Compiler


def test_compile_str_function_basic():
    code = """
def add(a, b):
    c = a + b
    return c
"""
    lua = Compiler(add_header=False).compile_str(code)
    expected = "function add(a, b)\n    local c = a + b\n    return c\nend\n"
    assert lua == expected


def test_header_metadata_generation(tmp_path: Path):
    src = tmp_path / "mod.py"
    src.write_text(
        "def mul(a, b):\n    d = a * b\n    return d\n",
        encoding="utf-8",
    )
    out = tmp_path / "mod.lua"
    Compiler(version="0.9.9", add_header=True).compile_file(src, out_path=out)

    text = out.read_text(encoding="utf-8")
    header = text.splitlines()[:8]
    header_str = "\n".join(header)

    assert "--  py2glua build metadata" in header_str
    assert "--  Version:       0.9.9" in header_str
    assert "--  Repository:" in header_str
    assert "--  Source:" in header_str
    assert "--  SHA1(src):" in header_str
    assert "--  SHA1(out):" in header_str
    assert "--  Build date:" in header_str


def test_compiled_function_body(tmp_path: Path):
    src = tmp_path / "mod.py"
    src.write_text(
        "def mul(a, b):\n    d = a * b\n    return d\n",
        encoding="utf-8",
    )
    out = tmp_path / "mod.lua"
    Compiler(version="0.9.9", add_header=True).compile_file(src, out_path=out)

    text = out.read_text(encoding="utf-8")

    assert "function mul(a, b)" in text
    assert "local d = a * b" in text
    assert "return d" in text
    assert text.strip().endswith("end")

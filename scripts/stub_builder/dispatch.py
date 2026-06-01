from pathlib import Path
from textwrap import dedent, indent

from generator import gen_class, gen_enum, gen_hook, gen_library, gen_panel, gen_struct

KIND_GENERATORS = {
    "library": gen_library,
    "class": gen_class,
    "panel": gen_panel,
    "struct": gen_struct,
    "enum": gen_enum,
    "hook": gen_hook,
}


def format_docstring(text: str, base_indent: int = 4) -> str:
    if not text:
        return ""
    clean = dedent(text).strip().replace('"""', '\\"\\"\\"')
    body = indent(clean, " " * base_indent)
    return f'"""{body}\n{" " * base_indent}"""'


def generate_category(name: str, data: dict) -> str:
    kind = str(data.get("kind"))
    gen_func = KIND_GENERATORS.get(kind, None)
    if gen_func:
        return gen_func(name, data)

    return f"# Unsupported kind: {kind} for {name}\n"


def generate_all(schema: dict, output_dir: Path):
    for name, entry in schema.items():
        code = generate_category(name, entry)
        fname = name.lower() + ".py"
        (output_dir / fname).write_text(code, encoding="utf-8")
        print(f"Сгенерирован {fname}")

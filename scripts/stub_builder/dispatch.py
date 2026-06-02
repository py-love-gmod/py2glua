from pathlib import Path

from generator import generate_class, generate_enum, generate_struct

KIND_GENERATORS = {
    "library": lambda name, data: generate_class(name, data, is_static=True),
    "libraryfunc": lambda name, data: generate_class(name, data, is_static=True),
    "class": lambda name, data: generate_class(name, data, is_static=False),
    "classfunc": lambda name, data: generate_class(name, data, is_static=False),
    "struct": generate_struct,
    "enum": generate_enum,
    # "panel": ...,
    # "panelfunc": ...,
    # "hook": ...,
}


def generate_category(name: str, data: dict) -> tuple[str, bool]:
    kind = str(data.get("kind"))
    gen_func = KIND_GENERATORS.get(kind)
    if gen_func:
        return gen_func(name, data), True

    return f"Unsupported kind: {kind} for {name}", False


def generate_all(schema: dict, output_dir: Path):
    for name, entry in schema.items():
        code, has_code = generate_category(name, entry)
        if has_code:
            kind = entry.get("kind", "unknown")
            fname = Path(f"{name.lower()}_{kind}.py")
            (output_dir / fname).write_text(code, encoding="utf-8")
            print(f"Сгенерирован {fname}")

        else:
            print(code)

from pathlib import Path
from typing import Dict, Set

from generator import generate_class, generate_enum, generate_struct
from import_collector import ImportCollector
from registry import TypeRegistry

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


def generate_category(name: str, data: dict) -> tuple[str, bool, set[str]]:
    kind = str(data.get("kind"))
    gen_func = KIND_GENERATORS.get(kind)
    if gen_func:
        code, used_types = gen_func(name, data)
        return code, True, used_types

    return f"Unsupported kind: {kind} for {name}", False, set()


def generate_init(output_dir: Path) -> None:
    init_path = output_dir / "__init__.py"
    lines = [
        '"""',
        "Автоматически сгенерированный пакет stub's для GLUA.",
        "Да поможет мне господь...",
        '"""',
        "",
    ]

    module_to_types: Dict[str, Set[str]] = {}
    for type_name in TypeRegistry.get_all_types():
        module = TypeRegistry.get_module_for_type(type_name)
        if module:
            module_to_types.setdefault(module, set()).add(type_name)

    for module, types in sorted(module_to_types.items()):
        types_sorted = sorted(types)
        lines.append(f"from .{module} import {', '.join(types_sorted)}")

    all_types = sorted(TypeRegistry.get_all_types())
    lines.extend(
        [
            "",
            f"__all__ = {repr(all_types)}",
        ]
    )
    init_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Сгенерирован {init_path}")


def generate_all(schema: dict, output_dir: Path):
    temp_storage: Dict[Path, tuple[str, Set[str], Dict]] = {}
    used_filenames: Set[str] = set()
    name_to_file_stem: Dict[str, str] = {}

    print("Первый проход: генерация файлов...")
    for name, entry in schema.items():
        name = name.replace(".", "_")
        code, has_code, used_types = generate_category(name, entry)

        if has_code and code:
            base_stem = name
            lower_stem = base_stem.lower()

            if lower_stem in used_filenames:
                counter = 1
                while True:
                    new_stem = f"{base_stem}_{counter}"
                    if new_stem.lower() not in used_filenames:
                        final_stem = new_stem
                        break

                    counter += 1

            else:
                final_stem = base_stem

            used_filenames.add(final_stem.lower())
            name_to_file_stem[name] = final_stem
            file_path = output_dir / f"{final_stem}.py"

            temp_storage[file_path] = (code, used_types, entry)

            defined_types = set()
            kind = entry.get("kind", "unknown")
            if kind in ["class", "library", "struct", "enum"]:
                defined_types.add(name)

            TypeRegistry.register(file_path, defined_types)

    print("Второй проход: добавление межфайловых импортов...")

    for file_path, (code, used_types, entry) in temp_storage.items():
        ImportCollector.reset()
        for t in used_types:
            ImportCollector.add_type(t)

        standard_imports = ImportCollector.get_imports()
        cross_imports = set()
        for t in used_types:
            imp = TypeRegistry.get_import_for_type(t, file_path)
            if imp:
                cross_imports.add(imp)

        final_lines = []
        if standard_imports:
            final_lines.extend(standard_imports)

        if cross_imports:
            if final_lines:
                final_lines.append("")

            final_lines.extend(sorted(cross_imports))

        if final_lines:
            final_lines.append("")

        final_lines.append(code)
        final_code = "\n".join(final_lines)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(final_code, encoding="utf-8")
        print(f"Сгенерирован {file_path.name}")

    for name, entry in schema.items():
        name = name.replace(".", "_")
        code, has_code, _ = generate_category(name, entry)
        if not has_code:
            print(code)

    generate_init(output_dir)

from pathlib import Path

from .codegen.base import GeneratedCode
from .codegen.class_gen import generate_class
from .codegen.enum_gen import generate_enum
from .codegen.struct_gen import generate_struct
from .function_filter import FunctionFilter
from .import_collector import resolve_standard_imports
from .import_registry import TypeRegistry
from .models import ClassDef, EnumDef, SchemaEntry, StructDef
from .schema_loader import load_entries
from .type_override import TypeOverride


def _unique_stem(name: str, used: set[str]) -> str:
    stem = name.lower()
    if stem not in used:
        return name

    counter = 1
    while True:
        new = f"{name}_{counter}"
        if new.lower() not in used:
            return new

        counter += 1


def generate_code_for_entry(
    entry: SchemaEntry,
    type_override: TypeOverride,
    function_filter: FunctionFilter,
) -> GeneratedCode:
    match entry:
        case ClassDef():
            return generate_class(entry, type_override, function_filter)

        case EnumDef():
            return generate_enum(entry)

        case StructDef():
            return generate_struct(entry, type_override)

        case _:
            return GeneratedCode("", set())


def generate_init(registry: TypeRegistry, output_dir: Path) -> None:
    lines = [
        '"""',
        "Auto-generated stubs package.",
        '"""',
        "",
    ]
    module_to_types: dict[str, set[str]] = {}
    for type_name in registry.all_types():
        module = registry.get_module_for_type(type_name)
        if module:
            module_to_types.setdefault(module, set()).add(type_name)

    for module, types in sorted(module_to_types.items()):
        types_sorted = sorted(types)
        lines.append(f"from .{module} import {', '.join(types_sorted)}")

    all_types = sorted(registry.all_types())
    lines.extend(
        [
            "",
            f"__all__ = {all_types!r}",
        ]
    )
    init_path = output_dir / "__init__.py"
    init_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {init_path}")


def generate_all(
    schema: dict[str, dict],
    output_dir: Path,
    type_override: TypeOverride,
    function_filter: FunctionFilter,
) -> None:
    entries = load_entries(schema)
    used_filenames: set[str] = set()
    registry = TypeRegistry()
    temp_storage: dict[Path, tuple[str, set[str], SchemaEntry]] = {}

    for name, entry in entries.items():
        gen = generate_code_for_entry(entry, type_override, function_filter)
        if not gen.code:
            print(f"Warning: no code generated for {name}")
            continue

        stem = _unique_stem(name, used_filenames)
        used_filenames.add(stem.lower())
        file_path = output_dir / f"{stem}.py"
        temp_storage[file_path] = (gen.code, gen.used_types, entry)

        if isinstance(entry, (ClassDef, EnumDef, StructDef)):
            registry.register_type(name, file_path)

    for file_path, (code, used_types, entry) in temp_storage.items():
        standard_imports = resolve_standard_imports(used_types)
        cross_imports: set[str] = set()
        for t in used_types:
            imp = registry.get_import_for_type(t, file_path)
            if imp:
                cross_imports.add(imp)

        final_lines = ["from __future__ import annotations", ""]
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
        print(f"Generated {file_path.name}")

    generate_init(registry, output_dir)

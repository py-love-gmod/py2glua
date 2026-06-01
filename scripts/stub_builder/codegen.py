import re
from pathlib import Path
from typing import List, Set, Tuple

BUILTIN_TYPES = {
    "string": "str",
    "number": "int | float",
    "boolean": "bool",
    "function": "Callable",
    "table": "dict",
    "nil": "None",
    "vararg": "Any",
    "any": "Any",
}


def normalize_type(gmod_type: str) -> Tuple[str, bool]:
    if not gmod_type:
        return "Any", False

    t = gmod_type.strip()

    if "|" in t:
        parts = [normalize_type(part)[0] for part in t.split("|")]
        return " | ".join(parts), False

    if "<" in t or "{" in t:
        return f'"{t}"', False

    lower = t.lower()
    if lower in BUILTIN_TYPES:
        return BUILTIN_TYPES[lower], False

    if t and t[0].isupper():
        return t, True

    return t, False


def _parse_args(args_block: List[dict]) -> List[dict]:
    result = []
    for group in args_block:
        for arg in group.get("args", []):
            if arg.get("name"):
                result.append(arg)

    return result


def _collect_typing_imports(annotation: str, used: Set[str]):
    for name in ("Any", "Callable", "Optional", "Union", "List", "Dict"):
        if re.search(r"\b" + name + r"\b", annotation):
            used.add(name)


def generate_module(
    entity_name: str, schema_entry: dict, known_entities: Set[str]
) -> str:
    lines = ["from __future__ import annotations"]
    typing_used = set()
    imports = set()

    description = schema_entry.get("description", "")
    methods = schema_entry.get("methods", [])

    class_name = (
        entity_name.capitalize()
        if entity_name and entity_name[0].islower()
        else entity_name
    )

    lines.append("")
    lines.append(f"class {class_name}:")
    if description:
        lines.append(f'    """\n    {description}\n    """')
    else:
        lines.append('    """..."""')

    for method in methods:
        name = method["name"]
        desc = method.get("description", "").strip()
        args_block = method.get("arguments", [])
        returns_block = method.get("returns", [])

        args = _parse_args(args_block)
        params = []
        for arg in args:
            arg_name = arg["name"]
            py_type, is_custom = normalize_type(arg.get("type", ""))
            if is_custom:
                if py_type in known_entities:
                    imports.add(py_type)
                    if arg_name == py_type:
                        arg_name = f"{arg_name}_param"
                else:
                    py_type = "Any"
            params.append(f"{arg_name}: {py_type}")
            _collect_typing_imports(py_type, typing_used)

        ret_types = []
        for ret in returns_block:
            py_type, is_custom = normalize_type(ret.get("type", ""))
            if is_custom:
                if py_type in known_entities:
                    imports.add(py_type)
                else:
                    py_type = "Any"
            ret_types.append(py_type)
            _collect_typing_imports(py_type, typing_used)

        if len(ret_types) == 1:
            return_annotation = ret_types[0]
        elif ret_types:
            return_annotation = f"tuple[{', '.join(ret_types)}]"
        else:
            return_annotation = "None"
        _collect_typing_imports(return_annotation, typing_used)

        lines.append("    @staticmethod")
        params_str = ", ".join(params)
        lines.append(f"    def {name}({params_str}) -> {return_annotation}:")
        lines.append(f'        """{desc}"""')
        lines.append("        ...")
        lines.append("")

    if typing_used:
        typing_line = "from typing import " + ", ".join(sorted(typing_used))
        lines.insert(1, typing_line)

    if imports:
        imports.discard(class_name)
        import_lines = sorted(f"from .{imp.lower()} import {imp}" for imp in imports)
        insert_pos = 2 if typing_used else 1
        for i, line in enumerate(import_lines):
            lines.insert(insert_pos + i, line)

    return "\n".join(lines)


def generate_all(schema: dict, output_dir: Path):
    """Создаёт .py файлы для всех сущностей."""
    known = set(schema.keys())
    for name, entry in schema.items():
        code = generate_module(name, entry, known)
        fname = name.lower() + ".py"
        (output_dir / fname).write_text(code, encoding="utf-8")
        print(f"Сгенерирован {fname}")

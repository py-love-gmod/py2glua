#!/usr/bin/env python3
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

ZIP_PATH = Path("scripts/2026-03-31_16-30-01.json.zip")
OUT_DIR = Path("bindings")
OUT_DIR.mkdir(exist_ok=True)

MANUAL_FILES = {"hooks.py", "realms.py"}

TYPE_MAP = {
    "number": "float | int",
    "string": "str",
    "boolean": "bool",
    "table": "dict",
    "function": "Callable",
    "nil": "nil_type",
    "Panel": "Panel",
    "Player": "Player",
    "Entity": "Entity",
    "Vehicle": "Vehicle",
    "Weapon": "Weapon",
    "Color": "Color",
    "Vector": "Vector",
    "Angle": "Angle",
    "Material": "Material",
    "Texture": "Texture",
    "RenderTarget": "RenderTarget",
    "IMaterial": "IMaterial",
    "IShader": "IShader",
    "Pixel": "Pixel",
    "userdata": "Any",
}
DEFAULT_TYPE = "Any"

REALM_VALID = {"shared", "client", "server", "menu"}
REALM_SKIP = {"and", "or", "of", "the", "for"}


def to_py_type(gmod_type: str) -> str:
    if not gmod_type:
        return DEFAULT_TYPE
    clean = gmod_type.replace("[[", "").replace("]]", "").strip()
    if clean.startswith("number"):
        return "float | int"
    if clean.startswith("string"):
        return "str"
    if clean.startswith("boolean"):
        return "bool"
    if clean.startswith("table"):
        return "dict"
    if clean.startswith("function"):
        return "Callable"
    if clean.startswith("nil"):
        return "nil_type"
    return TYPE_MAP.get(clean, clean)


def realm_list(realm_str):
    if not realm_str:
        return ["Realm.SHARED"]
    parts = realm_str.lower().split()
    out = []
    for p in parts:
        p_clean = p.strip(",.!?")
        if p_clean in REALM_VALID:
            out.append(f"Realm.{p_clean.upper()}")
        elif p_clean in REALM_SKIP:
            continue
    return out if out else ["Realm.SHARED"]


def make_decorator(orig_name, realm_str, method=False):
    rl = ", ".join(realm_list(realm_str))
    if method:
        short_name = orig_name.split(":")[-1] if ":" in orig_name else orig_name
        return f'@CompilerDirective.gmod_api("{short_name}", realm=[{rl}], method=True)'
    return f'@CompilerDirective.gmod_api("{orig_name}", realm=[{rl}])'


def escape_default(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value == "":
            return '""'
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def is_internal(desc: str) -> bool:
    """Возвращает True, если в описании есть **INTERNAL** (игнорируем регистр)."""
    if not desc:
        return False
    return "**internal**" in desc.lower()


def generate_function(orig_name, func_data, indent=0, is_method=False):
    args_list = []
    for arg_group in func_data.get("arguments", []):
        for arg in arg_group.get("args", []):
            args_list.append(arg)

    params = []
    vararg = False
    if is_method:
        params.append("self")

    for arg in args_list:
        arg_name = arg.get("name", "arg")
        if arg_name == "...":
            vararg = True
            continue
        arg_type = to_py_type(arg.get("type", ""))
        default = arg.get("default")
        if default is not None:
            default_repr = escape_default(default)
            if default_repr is not None:
                params.append(f"{arg_name}: {arg_type} = {default_repr}")
            else:
                params.append(f"{arg_name}: {arg_type}")
        else:
            params.append(f"{arg_name}: {arg_type}")

    if vararg:
        params.append("*args: Any")

    params_str = ", ".join(params)

    returns = func_data.get("returns", [])
    ret_type = to_py_type(returns[0].get("type", "")) if returns else "nil_type"

    desc = func_data.get("description", "")
    # Если INTERNAL – пропускаем (но на уровне элемента, а не функции – вызывающий код пропустит)
    indent_str = "    " * indent
    inner_indent = indent_str + "    "

    # Докстринг: без разрыва строк, просто вставляем как есть
    if desc:
        # Заменяем тройные кавычки внутри, чтобы не сломать синтаксис
        safe_desc = desc.replace('"""', '\\"\\"\\"')
        docstring = f'{inner_indent}"""{safe_desc}"""'
    else:
        docstring = f'{inner_indent}"""..."""'

    realm = func_data.get("realm", "shared")
    decorator = make_decorator(orig_name, realm, method=is_method)

    func_name = orig_name.split(":")[-1].split(".")[-1]

    return f"""{indent_str}{decorator}
{indent_str}def {func_name}({params_str}) -> {ret_type}:
{docstring}
{inner_indent}..."""


def generate_class(cls_name, methods_list):
    if cls_name.lower() == "table":
        return ""
    lines = [f"class {cls_name}:"]
    for meth in methods_list:
        if is_internal(meth.get("description", "")):
            continue
        meth_name = meth.get("name")
        if not meth_name:
            continue
        full_name = f"{cls_name}:{meth_name}"
        lines.append(generate_function(full_name, meth, indent=1, is_method=True))
    return "\n".join(lines)


def generate_library(lib_name, methods_list):
    if lib_name.lower() == "table":
        return ""
    lines = [f"class {lib_name}:"]
    for meth in methods_list:
        if is_internal(meth.get("description", "")):
            continue
        meth_name = meth.get("name")
        if not meth_name:
            continue
        full_name = f"{lib_name}.{meth_name}"
        lines.append(generate_function(full_name, meth, indent=1, is_method=False))
    return "\n".join(lines)


def generate_enum(enum_name, values_list):
    lines = [f"class {enum_name}(Enum):"]
    for v in values_list:
        name = v.get("name")
        value = v.get("value")
        if name and value is not None:
            lines.append(f"    {name} = {value}")
    return "\n".join(lines)


def generate_structure(struct_name, fields_list):
    lines = [f"class {struct_name}:"]
    for fld in fields_list:
        fld_name = fld.get("name")
        fld_type = to_py_type(fld.get("type", ""))
        lines.append(f"    {fld_name}: {fld_type}")
    return "\n".join(lines)


def load_and_group(zip_path: Path):
    libraries = defaultdict(list)
    classes = defaultdict(list)
    hooks = []
    enums = defaultdict(list)
    structures = defaultdict(list)
    game_events = defaultdict(list)
    panels = defaultdict(list)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        for json_file in tmp_path.rglob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                if not isinstance(items, list):
                    items = [items]

                for obj in items:
                    # Пропускаем INTERNAL на уровне объекта
                    if is_internal(obj.get("description", "")):
                        continue

                    obj_type = obj.get("type")
                    parent = obj.get("parent")
                    name = obj.get("name")
                    if not name:
                        continue

                    if obj_type == "libraryfunc":
                        lib_name = parent if parent else "unknown_lib"
                        libraries[lib_name].append(obj)
                    elif obj_type in ("classmethod", "classfunc", "class"):
                        class_name = parent if parent else "UnknownClass"
                        classes[class_name].append(obj)
                    elif obj_type == "hook":
                        hooks.append(obj)
                    elif obj_type == "enum":
                        enum_name = name
                        values = obj.get("values", [])
                        if values:
                            enums[enum_name].extend(values)
                    elif obj_type == "structure":
                        struct_name = name
                        fields = obj.get("fields", [])
                        if fields:
                            structures[struct_name].extend(fields)
                    elif obj_type == "gameevent":
                        event_name = name
                        fields = obj.get("fields", [])
                        if fields:
                            game_events[event_name].extend(fields)
                    elif obj_type == "panel":
                        panel_name = parent if parent else name
                        panels[panel_name].append(obj)
                    else:
                        if obj.get("values"):
                            enums[name].extend(obj.get("values", []))
                        elif obj.get("fields"):
                            structures[name].extend(obj.get("fields", []))
                        elif obj.get("arguments") is not None:
                            lib_name = parent if parent else "misc"
                            libraries[lib_name].append(obj)

    libs_list = [{"name": k, "methods": v} for k, v in libraries.items()]
    classes_list = [{"name": k, "methods": v} for k, v in classes.items()]
    enums_list = [{"name": k, "values": v} for k, v in enums.items()]
    structs_list = [{"name": k, "fields": v} for k, v in structures.items()]
    events_list = [{"name": k, "fields": v} for k, v in game_events.items()]
    panels_list = [{"name": k, "methods": v} for k, v in panels.items()]

    all_classes = classes_list + panels_list

    # Дедубликация хуков по имени
    unique_hooks = {}
    for h in hooks:
        h_name = h.get("name")
        if h_name and h_name not in unique_hooks:
            unique_hooks[h_name] = h
    hooks_dedup = list(unique_hooks.values())

    return {
        "Libraries": libs_list,
        "Classes": all_classes,
        "Hooks": hooks_dedup,
        "Enumerations": enums_list,
        "Structures": structs_list,
        "Game Events": events_list,
    }


def main():
    if not ZIP_PATH.exists():
        print(f"Ошибка: архив {ZIP_PATH} не найден.")
        return

    print("Загрузка и группировка данных из архива...")
    data = load_and_group(ZIP_PATH)
    print("Сгруппировано:")
    for k, v in data.items():
        print(f"  {k}: {len(v)} записей")

    # realms.py (не перезаписываем)
    realms_path = OUT_DIR / "realms.py"
    if not realms_path.exists():
        realms_path.write_text(
            """from enum import Enum

class Realm(Enum):
    CLIENT = "CLIENT"
    SERVER = "SERVER"
    SHARED = "SHARED"
    MENU = "MENU"
""",
            encoding="utf-8",
        )
        print("Generated realms.py")

    # Hooks -> HookType с docstring
    hooks_data = data.get("Hooks", [])
    if hooks_data:
        hook_lines = []
        for h in hooks_data:
            name = h.get("name")
            if not name:
                continue
            safe_name = name.replace(" ", "_")
            desc = h.get("description", "")
            # Экранируем тройные кавычки внутри описания
            if desc:
                desc = desc.replace('"""', '\\"\\"\\"')
            hook_lines.append(f"    {safe_name} = {repr(name)}")
            if desc:
                hook_lines.append(f'    """{desc}"""')
            else:
                hook_lines.append('    """..."""')
        hook_types_code = f"""from enum import Enum

class HookType(Enum):
{chr(10).join(hook_lines)}
"""
        (OUT_DIR / "hook_types.py").write_text(hook_types_code, encoding="utf-8")
        print("Generated hook_types.py")

    # Libraries
    libs_data = data.get("Libraries", [])
    if libs_data:
        out = []
        for lib in libs_data:
            lib_name = lib.get("name")
            methods = lib.get("methods", [])
            if lib_name and methods and lib_name.lower() != "table":
                out.append(generate_library(lib_name, methods))
        if out:
            (OUT_DIR / "libraries.py").write_text("\n\n".join(out), encoding="utf-8")
            print("Generated libraries.py")

    # Classes
    classes_data = data.get("Classes", [])
    if classes_data:
        out = []
        for cls in classes_data:
            cls_name = cls.get("name")
            methods = cls.get("methods", [])
            if cls_name and methods and cls_name.lower() != "table":
                out.append(generate_class(cls_name, methods))
        if out:
            (OUT_DIR / "classes.py").write_text("\n\n".join(out), encoding="utf-8")
            print("Generated classes.py")

    # Enumerations
    enums_data = data.get("Enumerations", [])
    if enums_data:
        out = []
        for enum in enums_data:
            enum_name = enum.get("name")
            values = enum.get("values", [])
            if enum_name and values:
                out.append(generate_enum(enum_name, values))
        if out:
            (OUT_DIR / "enums.py").write_text("\n\n".join(out), encoding="utf-8")
            print("Generated enums.py")

    # Structures
    structs_data = data.get("Structures", [])
    if structs_data:
        out = []
        for struct in structs_data:
            struct_name = struct.get("name")
            fields = struct.get("fields", [])
            if struct_name and fields:
                out.append(generate_structure(struct_name, fields))
        if out:
            (OUT_DIR / "structures.py").write_text("\n\n".join(out), encoding="utf-8")
            print("Generated structures.py")

    # Game Events
    events_data = data.get("Game Events", [])
    if events_data:
        out = []
        for ev in events_data:
            ev_name = ev.get("name")
            fields = ev.get("fields", [])
            if ev_name and fields:
                out.append(generate_structure(ev_name, fields))
        if out:
            (OUT_DIR / "gameevents.py").write_text("\n\n".join(out), encoding="utf-8")
            print("Generated gameevents.py")

    # __init__.py
    init_py = OUT_DIR / "__init__.py"
    if not init_py.exists():
        init_py.write_text(
            """from . import realms
from . import hook_types
from . import classes
from . import libraries
from . import enums
from . import structures
from . import gameevents
""",
            encoding="utf-8",
        )
        print("Generated __init__.py")

    print("Готово.")


if __name__ == "__main__":
    main()

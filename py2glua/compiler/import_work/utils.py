from __future__ import annotations

from pathlib import Path

from plg_reader import IRAttribute, IRImport, IRName, IRNode


def parse_name_entry(entry: str | tuple) -> tuple[str, str]:
    """Возвращает (оригинальное_имя, используемое_имя)."""
    if isinstance(entry, str):
        return entry, entry

    return entry[0], entry[1]


def build_attr_chain(parts: list[str], pos: tuple[int, int]) -> IRNode:
    """Строит цепочку IRAttribute из списка частей имени."""
    if not parts:
        raise ValueError("Пустая цепочка атрибутов")

    node: IRNode = IRName(pos=pos, name=parts[0])
    for attr in parts[1:]:
        node = IRAttribute(pos=pos, value=node, attr=attr)

    return node


def module_name_from_relative_path(rel_path: str) -> str:
    """Преобразует относительный путь в имя модуля: 'pkg/mod.py' -> 'pkg.mod'."""
    parts = Path(rel_path).parts
    name = parts[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]

    parts = list(parts[:-1]) + [name]
    if parts and parts[-1] == "__init__":
        parts.pop()

    return ".".join(parts)


def resolve_target_module(imp: IRImport, current_module: str) -> str:
    """Возвращает абсолютное имя модуля, на которое ссылается импорт."""
    module_parts = current_module.split(".") if current_module else []

    if imp.level == 0:
        return ".".join(imp.modules)

    if imp.level > len(module_parts):
        raise ValueError(
            f"Слишком глубокий относительный импорт (level={imp.level}) "
            f"в модуле {current_module}"
        )

    base = ".".join(module_parts[: -imp.level]) if imp.level < len(module_parts) else ""
    if imp.modules:
        return f"{base}.{'.'.join(imp.modules)}" if base else ".".join(imp.modules)

    return base

from __future__ import annotations

from concurrent.futures import as_completed
from pathlib import Path

from plg_reader import (
    IRAnnotatedAssign,
    IRAssign,
    IRAttribute,
    IRFile,
    IRFor,
    IRImport,
    IRName,
    IRNode,
    get_cpus_and_executor,
)


class ImportResolver:
    @staticmethod
    def _parse_name_entry(entry: str | tuple) -> tuple[str, str]:
        """
        Возвращает (оригинальное_имя, используемое_имя) из записи импорта.
        """
        if isinstance(entry, str):
            return entry, entry

        else:
            return entry[0], entry[1]

    @staticmethod
    def _build_attr_chain(parts: list[str], pos: tuple[int, int] = (0, 0)) -> IRNode:
        """
        Строит цепочку атрибутов: a.b.c -> IRAttribute(IRAttribute(IRName('a'), 'b'), 'c')
        """
        if not parts:
            raise ValueError("Пустая цепочка атрибутов")

        node = IRName(pos=pos, name=parts[0])
        for attr in parts[1:]:
            node = IRAttribute(pos=pos, value=node, attr=attr)

        return node

    @staticmethod
    def _get_module_name(file_path: Path) -> str:
        parts = list(file_path.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        if parts and parts[-1] == "__init__":
            parts.pop()

        return ".".join(parts)

    @classmethod
    def _replace_names_in_node(
        cls,
        node: IRNode,
        mapping: dict[str, IRNode],
        is_target: bool = False,
    ) -> IRNode:
        """
        Рекурсивно заменяет IRName в выражениях согласно mapping.
        Не заменяет имена в определяющих позициях (targets в присваиваниях и т.п.).
        """
        if isinstance(node, IRName) and not is_target:
            if node.name in mapping:
                new_node = mapping[node.name]
                new_node.pos = node.pos
                return new_node

        if isinstance(node, IRAssign):
            node.targets = [
                cls._replace_names_in_node(t, mapping, is_target=True)
                for t in node.targets
            ]
            node.value = cls._replace_names_in_node(
                node.value, mapping, is_target=False
            )
            return node

        if isinstance(node, IRAnnotatedAssign):
            node.target = cls._replace_names_in_node(
                node.target, mapping, is_target=True
            )
            if node.annotation is not None:
                node.annotation = cls._replace_names_in_node(
                    node.annotation, mapping, is_target=False
                )

            if node.value is not None:
                node.value = cls._replace_names_in_node(
                    node.value, mapping, is_target=False
                )

            return node

        if isinstance(node, IRFor):
            node.target = cls._replace_names_in_node(
                node.target, mapping, is_target=True
            )
            node.iter = cls._replace_names_in_node(node.iter, mapping, is_target=False)
            node.body = [
                cls._replace_names_in_node(stmt, mapping, is_target=False)
                for stmt in node.body
            ]
            return node

        for field_name in node.__dataclass_fields__:
            if field_name == "pos":
                continue

            value = getattr(node, field_name)
            if isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, IRNode):
                        new_list.append(
                            cls._replace_names_in_node(item, mapping, is_target=False)
                        )

                    else:
                        new_list.append(item)

                setattr(node, field_name, new_list)

            elif isinstance(value, dict):
                new_dict = {}
                for k, v in value.items():
                    if isinstance(v, IRNode):
                        new_dict[k] = cls._replace_names_in_node(
                            v, mapping, is_target=False
                        )

                    else:
                        new_dict[k] = v

                setattr(node, field_name, new_dict)

            elif isinstance(value, IRNode):
                setattr(
                    node,
                    field_name,
                    cls._replace_names_in_node(value, mapping, is_target=False),
                )

        return node

    @classmethod
    def _resolve_import(cls, ir_file: IRFile) -> IRFile:
        """
        Преобразует все импорты в файле:
        1. Все импорты становятся `import module` (без from, без алиасов).
        2. Использования импортированных имён заменяются полными атрибутными цепочками.
        """
        current_module = cls._get_module_name(ir_file.path)
        module_parts = current_module.split(".") if current_module else []
        mapping: dict[str, IRNode] = {}
        modules_to_import: set[str] = set()
        for imp in ir_file.imports:
            if imp.level == 0:
                base_module = ".".join(imp.modules)

            else:
                if imp.level > len(module_parts):
                    raise ValueError(
                        f"Слишком глубокий относительный импорт (level={imp.level}) "
                        f"в модуле {current_module}"
                    )

                base = (
                    ".".join(module_parts[: -imp.level])
                    if imp.level < len(module_parts)
                    else ""
                )
                if imp.modules:
                    base_module = (
                        f"{base}.{'.'.join(imp.modules)}"
                        if base
                        else ".".join(imp.modules)
                    )

                else:
                    base_module = base

            modules_to_import.add(base_module)
            if imp.is_from:
                for entry in imp.names:
                    orig, alias = cls._parse_name_entry(entry)
                    full_parts = base_module.split(".") + [orig]
                    chain = cls._build_attr_chain(full_parts)
                    mapping[alias] = chain

            else:
                if imp.names:
                    for entry in imp.names:
                        orig, alias = cls._parse_name_entry(entry)
                        chain = cls._build_attr_chain(base_module.split("."))
                        mapping[alias] = chain

        new_imports = []
        for mod in sorted(modules_to_import):
            new_imports.append(
                IRImport(pos=(1, 0), modules=[mod], names=[], is_from=False, level=0)
            )

        ir_file.imports = new_imports
        ir_file.body = [
            cls._replace_names_in_node(stmt, mapping) for stmt in ir_file.body
        ]

        return ir_file

    @classmethod
    def resolve_imports(cls, irs: dict[str, IRFile]) -> dict[str, IRFile]:
        if len(irs) == 1:
            path, ir_file = next(iter(irs.items()))
            return {path: cls._resolve_import(ir_file)}

        cpus, Executor = get_cpus_and_executor()
        result: dict[str, IRFile] = {}
        with Executor(max_workers=cpus) as executor:
            future_to_path = {
                executor.submit(cls._resolve_import, ir_file): path
                for path, ir_file in irs.items()
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result[path] = future.result()

                except Exception as exc:
                    raise RuntimeError(
                        f"Ошибка при разрешении импортов в {path}: {exc}"
                    ) from exc

        return result

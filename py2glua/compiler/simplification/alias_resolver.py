import dataclasses
from pathlib import Path
from typing import Any

from plg_reader import (
    IRAttribute,
    IRFile,
    IRImport,
    IRName,
    IRNode,
)
from utils import Config


class AliasResolver:
    @classmethod
    def resolve(cls, ir_file: IRFile) -> IRFile:
        root_path = Config.get_path_input()
        mapping, needed_imports = cls._build_mapping_and_imports(ir_file, root_path)
        new_ir = cls._transform(ir_file, mapping)
        new_ir.imports = list(needed_imports.values())
        return new_ir

    @classmethod
    def _build_mapping_and_imports(cls, ir_file: IRFile, root_path: Path):
        mapping = {}
        imports = {}
        current_path = ir_file.path

        for imp in ir_file.imports:
            if imp.is_from:
                if imp.level > 0:
                    base = current_path.parent
                    for _ in range(imp.level):
                        base = base.parent

                    try:
                        rel = base.relative_to(root_path)
                        module_parts = list(rel.parts)

                    except ValueError:
                        continue

                else:
                    module_parts = imp.modules[0].split(".") if imp.modules else []

                if not module_parts:
                    continue

                module_name = ".".join(module_parts)
                if module_name not in imports:
                    imports[module_name] = IRImport(
                        pos=imp.pos,
                        modules=[module_name],
                        names=[module_name],
                        is_from=False,
                        level=0,
                    )

                for name_item in imp.names:
                    if isinstance(name_item, tuple):
                        original, alias = name_item

                    else:
                        original = name_item
                        alias = name_item

                    full_name = f"{module_name}.{original}"
                    mapping[alias] = full_name

            else:
                for mod, name_item in zip(imp.modules, imp.names):
                    if isinstance(name_item, tuple):
                        original, alias = name_item

                    else:
                        alias = name_item

                    if alias == mod:
                        if mod not in imports:
                            imports[mod] = imp

                        continue

                    if mod not in imports:
                        imports[mod] = IRImport(
                            pos=imp.pos,
                            modules=[mod],
                            names=[mod],
                            is_from=False,
                            level=0,
                        )
                    mapping[alias] = mod

        return mapping, imports

    @classmethod
    def _transform(cls, node: Any, mapping: dict[str, str]) -> Any:
        if isinstance(node, IRName):
            if node.name in mapping:
                new_name = mapping[node.name]
                return cls._make_attribute_chain(node.pos, new_name.split("."))

            return node

        elif dataclasses.is_dataclass(node):
            new_kwargs = {}
            for field in dataclasses.fields(node):
                value = getattr(node, field.name)
                new_kwargs[field.name] = cls._transform(value, mapping)

            return type(node)(**new_kwargs)

        elif isinstance(node, list):
            return [cls._transform(item, mapping) for item in node]

        elif isinstance(node, dict):
            return {k: cls._transform(v, mapping) for k, v in node.items()}

        else:
            return node

    @staticmethod
    def _make_attribute_chain(pos: tuple[int, int], parts: list[str]) -> IRNode:
        node = IRName(pos=pos, name=parts[0])
        for part in parts[1:]:
            node = IRAttribute(pos=pos, value=node, attr=part)

        return node

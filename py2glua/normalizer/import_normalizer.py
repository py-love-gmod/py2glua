import re
import sys
from pathlib import Path

from ..config import Py2GluaConfig
from ..ir.ir_base import (
    Attribute,
    ClassDef,
    File,
    FunctionDef,
    Import,
    ImportType,
    IRNode,
    VarLoad,
)
from .base_normalizer import BaseNormalizer


class ImportNormalizer(BaseNormalizer):
    priority = 1

    @classmethod
    def run(cls, file: File) -> None:
        import_map = cls._collect_imports_and_tag_types(file)
        if not import_map:
            return

        file.meta["import_map"] = import_map
        try:
            cls._apply_import_map(file, import_map)

        finally:
            file.meta.pop("import_map", None)

    @classmethod
    def _collect_imports_and_tag_types(cls, file: File) -> dict[str, str]:
        import_map: dict[str, str] = {}

        for node in file.walk():
            if not isinstance(node, Import):
                continue

            cls._set_import_type(node)

            if node.module and node.names:
                module = node.module
                for name, alias in zip(node.names, node.aliases):
                    full = f"{module}.{name}"
                    import_map[name] = full
                    if alias:
                        import_map[alias] = full

                node.module = None
                node.names = [f"{module}.{n}" for n in node.names]
                node.aliases = [None] * len(node.names)
                continue

            if node.module is None and node.names:
                for name, alias in zip(node.names, node.aliases):
                    if alias:
                        import_map[alias] = name

        return import_map

    @staticmethod
    def _set_import_type(node: Import) -> None:
        module = node.module or (node.names[0] if node.names else None)
        if not module:
            node.import_type = ImportType.UNKNOWN
            return

        if module in sys.stdlib_module_names:
            node.import_type = ImportType.PYTHON_STD
            return

        if module.startswith("py2glua"):
            node.import_type = ImportType.INTERNAL
            return

        src_root: Path = Py2GluaConfig.data["build"]["source"]
        mod_path = src_root / module.replace(".", "/")
        if (mod_path.with_suffix(".py")).exists() or (
            mod_path / "__init__.py"
        ).exists():
            node.import_type = ImportType.LOCAL
            return

        node.import_type = ImportType.EXTERNAL

    @classmethod
    def _apply_import_map(cls, file: File, import_map: dict[str, str]) -> None:
        for node in file.walk():
            if isinstance(node, VarLoad) and node.name in import_map:
                qualified = import_map[node.name]
                chain_top = cls._build_attr_chain(qualified, node)
                cls._replace_node(node, chain_top)

        for node in file.walk():
            if isinstance(node, (FunctionDef, ClassDef)):
                if not getattr(node, "decorators", None):
                    continue
                node.decorators = [
                    cls._qualify_decorator_string(d, import_map)
                    if isinstance(d, str)
                    else d
                    for d in node.decorators
                ]

    @staticmethod
    def _build_attr_chain(qualified: str, template: IRNode) -> IRNode:
        parts = qualified.split(".")
        root: IRNode = VarLoad(
            lineno=template.lineno,
            col_offset=template.col_offset,
            file=template.file,
            parent=None,
            name=parts[0],
        )
        cur = root
        for attr in parts[1:]:
            nxt = Attribute(
                lineno=template.lineno,
                col_offset=template.col_offset,
                file=template.file,
                parent=None,
                value=cur,
                attr=attr,
            )
            cur.parent = nxt
            cur = nxt

        return cur

    @classmethod
    def _replace_node(cls, old: IRNode, new: IRNode) -> None:
        parent = old.parent
        if parent is None:
            return

        replaced = cls._replace_in_object_attr(parent, old, new)
        if replaced:
            new.parent = parent
            return

        replaced = cls._replace_in_object_list(parent, old, new)
        if replaced:
            new.parent = parent
            return

    @staticmethod
    def _replace_in_object_attr(obj: IRNode, old: IRNode, new: IRNode) -> bool:
        for k, v in vars(obj).items():
            if k in {"parent", "file"}:
                continue

            if v is old:
                setattr(obj, k, new)
                return True

        return False

    @staticmethod
    def _replace_in_object_list(obj: IRNode, old: IRNode, new: IRNode) -> bool:
        changed = False
        for k, v in vars(obj).items():
            if k in {"parent", "file"}:
                continue

            if isinstance(v, list):
                for i, it in enumerate(v):
                    if it is old:
                        v[i] = new
                        changed = True
        return changed

    @staticmethod
    def _qualify_decorator_string(src: str, import_map: dict[str, str]) -> str:
        if not import_map:
            return src

        result = src
        for short, full in sorted(import_map.items(), key=lambda kv: -len(kv[0])):
            pattern = rf"(?<!\.)\b{re.escape(short)}\b"
            result = re.sub(pattern, full, result)

        return result

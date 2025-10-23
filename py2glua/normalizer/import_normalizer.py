import sys
from pathlib import Path

from ..config import Py2GluaConfig
from ..ir.ir_base import (
    Attribute,
    BinOp,
    BoolOp,
    Call,
    ClassDef,
    Compare,
    DictLiteral,
    ExceptHandler,
    File,
    For,
    FunctionDef,
    If,
    Import,
    ImportType,
    IRNode,
    ListLiteral,
    Return,
    Subscript,
    Try,
    TupleLiteral,
    UnaryOp,
    VarLoad,
    VarStore,
    While,
    With,
)
from .base_normalizer import BaseNormalizer


class ImportNormalizer(BaseNormalizer):
    priority = 1

    @classmethod
    def run(cls, file: File) -> None:
        file.meta["import_map"] = {}

        for node in file.walk():
            if isinstance(node, Import):
                cls._normilize_import(node, file)

        for node in file.walk():
            if not isinstance(node, Import):
                cls._normilize_nodes(node, file)

        del file.meta["import_map"]

    @classmethod
    def _set_import_type(cls, node: Import):
        module = node.module
        if module is None:
            module = node.names[0] if node.names else None
            if module is None:
                node.import_type = ImportType.UNKNOWN
                return

        if module in sys.stdlib_module_names:
            node.import_type = ImportType.PYTHON_STD
            return

        if module.startswith("py2glua"):
            node.import_type = ImportType.INTERNAL
            return

        source_root: Path = Py2GluaConfig.data["build"]["source"]
        mod_path = source_root / module.replace(".", "/")
        candidate_files = [
            mod_path.with_suffix(".py"),
            mod_path / "__init__.py",
        ]
        if any(p.exists() for p in candidate_files):
            node.import_type = ImportType.LOCAL
            return

        node.import_type = ImportType.EXTERNAL

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

    @staticmethod
    def _replace_in_parent(parent: IRNode | None, old: IRNode, new: IRNode) -> None:
        if parent is None:
            return

        def repl_in_list(items: list[IRNode]) -> bool:
            for i, it in enumerate(items):
                if it is old:
                    items[i] = new
                    new.parent = parent
                    return True

            return False

        if isinstance(parent, VarStore):
            if parent.value is old:
                parent.value = new
                new.parent = parent
                return

        if isinstance(parent, Call):
            if parent.func is old:
                parent.func = new
                new.parent = parent
                return

            if repl_in_list(parent.args):
                return

        if isinstance(parent, Attribute):
            if parent.value is old:
                parent.value = new
                new.parent = parent
                return

        if isinstance(parent, Subscript):
            if parent.value is old:
                parent.value = new
                new.parent = parent
                return

            if parent.index is old:
                parent.index = new
                new.parent = parent
                return

        if isinstance(parent, UnaryOp):
            if parent.operand is old:
                parent.operand = new
                new.parent = parent
                return

        if isinstance(parent, BinOp):
            if parent.left is old:
                parent.left = new
                new.parent = parent
                return

            if parent.right is old:
                parent.right = new
                new.parent = parent
                return

        if isinstance(parent, BoolOp):
            if parent.left is old:
                parent.left = new
                new.parent = parent
                return

            if parent.right is old:
                parent.right = new
                new.parent = parent
                return

        if isinstance(parent, Compare):
            if parent.left is old:
                parent.left = new
                new.parent = parent
                return

            if parent.right is old:
                parent.right = new
                new.parent = parent
                return

        if isinstance(parent, Return):
            if parent.value is old:
                parent.value = new
                new.parent = parent
                return

        if isinstance(parent, FunctionDef):
            if repl_in_list(parent.body):
                return

        if isinstance(parent, ClassDef):
            if repl_in_list(parent.bases):
                return

            if repl_in_list(parent.body):
                return

        if isinstance(parent, If):
            if parent.test is old:
                parent.test = new
                new.parent = parent
                return

            if repl_in_list(parent.body):
                return

            if repl_in_list(parent.orelse):
                return

        if isinstance(parent, While):
            if parent.test is old:
                parent.test = new
                new.parent = parent
                return

            if repl_in_list(parent.body):
                return

            if repl_in_list(parent.orelse):
                return

        if isinstance(parent, For):
            if parent.target is old:
                parent.target = new
                new.parent = parent
                return

            if parent.iter is old:
                parent.iter = new
                new.parent = parent
                return

            if repl_in_list(parent.body):
                return

            if repl_in_list(parent.orelse):
                return

        if isinstance(parent, With):
            if parent.context is old:
                parent.context = new
                new.parent = parent
                return

            if parent.target is old:
                parent.target = new
                new.parent = parent
                return

            if repl_in_list(parent.body):
                return

        if isinstance(parent, ExceptHandler):
            if parent.type is old:
                parent.type = new
                new.parent = parent
                return

            if repl_in_list(parent.body):
                return

        if isinstance(parent, Try):
            if repl_in_list(parent.body):
                return

            if repl_in_list(parent.orelse):
                return

            if repl_in_list(parent.finalbody):
                return

        if isinstance(parent, ListLiteral):
            if repl_in_list(parent.elements):
                return

        if isinstance(parent, TupleLiteral):
            if repl_in_list(parent.elements):
                return

        if isinstance(parent, DictLiteral):
            if repl_in_list(parent.keys):
                return

            if repl_in_list(parent.values):
                return

        if isinstance(parent, File):
            if repl_in_list(parent.body):
                return

    @classmethod
    def _normilize_import(cls, node: Import, file: File):
        cls._set_import_type(node)

        import_map: dict[str, str] = file.meta["import_map"]

        if node.module is not None and node.names:
            new_names: list[str] = []
            new_aliases: list[str | None] = []

            for name, alias in zip(node.names, node.aliases):
                full = f"{node.module}.{name}"
                import_map[name] = full
                if alias:
                    import_map[alias] = full

                new_names.append(full)
                new_aliases.append(None)

            node.module = None
            node.names = new_names
            node.aliases = new_aliases
            return

        if node.module is None and node.names:
            for name, alias in zip(node.names, node.aliases):
                if alias:
                    import_map[alias] = name

            return

    @classmethod
    def _normilize_nodes(cls, node: IRNode, file: File):
        import_map: dict[str, str] = file.meta.get("import_map", {})
        if not import_map:
            return

        if isinstance(node, VarLoad) and node.name in import_map:
            qualified = import_map[node.name]
            chain_top = cls._build_attr_chain(qualified, node)
            parent = node.parent
            cls._replace_in_parent(parent, node, chain_top)

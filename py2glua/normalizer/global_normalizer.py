from ..ir.ir_base import Attribute, Call, ClassDef, File, FunctionDef, VarLoad, VarStore
from .base_normalizer import BaseNormalizer


class GlobalNormalizer(BaseNormalizer):
    priority: int = 2

    @classmethod
    def run(cls, file: File) -> None:
        globals_table: dict[str, dict] = {}

        for node in file.walk():
            if isinstance(node, VarStore):
                if isinstance(node.value, Call):
                    if cls._is_global_var_call(node.value):
                        globals_table[node.name] = cls._extract_var_meta(
                            node.value, node
                        )
                        if node.file is None:
                            raise

            if isinstance(node, (FunctionDef, ClassDef)):
                if getattr(node, "decorators", None):
                    for deco in node.decorators:
                        if isinstance(deco, str) and cls._is_global_mark_decorator(
                            deco
                        ):
                            meta = cls._extract_mark_meta(deco, node)
                            globals_table[node.name] = meta
                            if node.file is None:
                                raise

        if globals_table:
            file.meta["globals"] = globals_table

    @staticmethod
    def _is_global_var_call(call: Call) -> bool:
        func = call.func
        if not isinstance(func, Attribute):
            return False

        if func.attr != "var":
            return False

        value = func.value
        if isinstance(value, VarLoad) and value.name == "Global":
            return True

        if isinstance(value, Attribute) and isinstance(value.value, VarLoad):
            return value.value.name == "py2glua" and value.attr == "Global"

        return False

    @staticmethod
    def _is_global_mark_decorator(deco: str) -> bool:
        return deco.startswith("Global.mark") or deco.startswith("py2glua.Global.mark")

    @staticmethod
    def _extract_var_meta(call: Call, node: VarStore) -> dict:
        external = False

        if call.kwargs.get("external", False):
            external = True

        return {
            "name": node.name,
            "external": external,
            "type": "var",
            "lineno": node.lineno,
            "col_offset": node.col_offset,
        }

    @staticmethod
    def _extract_mark_meta(deco: str, node) -> dict:
        external = False
        if "external=True" in deco:
            external = True

        return {
            "name": node.name,
            "external": external,
            "type": "function" if node.__class__.__name__ == "FunctionDef" else "class",
            "lineno": node.lineno,
            "col_offset": node.col_offset,
        }

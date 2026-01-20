from __future__ import annotations

from ...py.ir_dataclass import PyIRClassDef, PyIRFile, PyIRNode


class RemoveNamedClassPass:
    _TARGET_CLASSES: set[str] = {
        "CompilerDirective",
        "InternalCompilerDirective",
    }

    @classmethod
    def run(cls, files: list[PyIRFile]) -> list[PyIRFile]:
        for f in files:
            if f.body:
                f.body = cls._filter_body(f.body)

        return files

    @classmethod
    def _filter_body(cls, body: list[PyIRNode]) -> list[PyIRNode]:
        out: list[PyIRNode] = []

        for node in body:
            if isinstance(node, PyIRClassDef):
                if cls._is_target_class(node):
                    continue

                else:
                    node.body = cls._filter_body(node.body)
                    out.append(node)
            else:
                out.append(node)

        return out

    @classmethod
    def _is_target_class(cls, node: PyIRClassDef) -> bool:
        return node.name in cls._TARGET_CLASSES

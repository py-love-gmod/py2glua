from __future__ import annotations

from ...py.ir_dataclass import PyIRClassDef, PyIRFile, PyIRNode


class RemoveDecoratedClassPass:
    _TARGET_DECORATORS: set[str] = {
        "CompilerDirective.no_compile",
        "CompilerDirective.gmod_special_enum",  # TODO: Это не должно происходить тут. Просто ибо бред это
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
                if cls._has_target_decorator(node):
                    continue

                else:
                    node.body = cls._filter_body(node.body)
                    out.append(node)
            else:
                out.append(node)

        return out

    @classmethod
    def _has_target_decorator(cls, node: PyIRClassDef) -> bool:
        decs = node.decorators
        if not decs:
            return False

        for dec in decs:
            if dec.name in cls._TARGET_DECORATORS:
                return True

        return False

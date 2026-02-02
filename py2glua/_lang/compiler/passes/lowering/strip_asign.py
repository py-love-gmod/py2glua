from __future__ import annotations

from ....compiler.passes.analysis.symlinks import SymLinkContext
from ....py.ir_dataclass import (
    PyIRAnnotatedAssign,
    PyIRAnnotation,
    PyIRAssign,
    PyIRAugAssign,
    PyIRClassDef,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRReturn,
    PyIRWhile,
    PyIRWith,
)


class StripPythonOnlyNodesPass:
    """
    Удаляет Python-only ноды, которые не должны доходить до emit:
      - PyIRAnnotation (x: T)
      - PyIRAnnotatedAssign (x: T = expr) -> превращаем в обычный Assign
    """

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        _ = ctx

        def rw_list(stmts: list[PyIRNode]) -> list[PyIRNode]:
            out: list[PyIRNode] = []
            for s in stmts:
                out.extend(rw_stmt(s))

            return out

        def rw_stmt(node: PyIRNode) -> list[PyIRNode]:
            if isinstance(node, PyIRAnnotation):
                return []

            if isinstance(node, PyIRAnnotatedAssign):
                return [
                    PyIRAssign(
                        line=node.line,
                        offset=node.offset,
                        targets=[],
                        value=node.value,
                    )
                ]

            if isinstance(node, PyIRIf):
                node.body = rw_list(node.body)
                node.orelse = rw_list(node.orelse)
                return [node]

            if isinstance(node, PyIRWhile):
                node.body = rw_list(node.body)
                return [node]

            if isinstance(node, PyIRFor):
                node.body = rw_list(node.body)
                return [node]

            if isinstance(node, PyIRWith):
                node.body = rw_list(node.body)
                return [node]

            if isinstance(node, PyIRReturn):
                return [node]

            if isinstance(node, PyIRAugAssign):
                return [node]

            if isinstance(node, PyIRFunctionDef):
                node.body = rw_list(node.body)
                return [node]

            if isinstance(node, PyIRClassDef):
                node.body = rw_list(node.body)
                return [node]

            return [node]

        ir.body = rw_list(ir.body)
        return ir

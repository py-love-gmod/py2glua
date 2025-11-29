from pathlib import Path

from ..parse import (
    PyLogicBlockBuilder,
    PyLogicKind,
    PyLogicNode,
)
from .builders import ImportBuilder
from .ir_dataclass import PyIRContext, PyIRFile, PyIRNode


class PyIRBuilder:
    @classmethod
    def build_file(cls, source: str, path_to_file: Path | None = None) -> PyIRFile:
        logic_blocks = PyLogicBlockBuilder.build(source)

        context = PyIRContext(
            parent_context=None,
            meta={
                "aliases": {},
                "imports": [],
                "__file__": path_to_file,
            },
        )

        py_ir_file = PyIRFile(
            line=None,
            offset=None,
            path=path_to_file,
            context=context,
        )
        py_ir_file.body = cls._build_ir_block(logic_blocks, py_ir_file)
        return py_ir_file

    # region Core block dispatcher
    @classmethod
    def _build_ir_block(
        cls,
        nodes: list[PyLogicNode],
        parent_obj: PyIRNode,
    ) -> list[PyIRNode]:
        dispatch = {
            PyLogicKind.FUNCTION: None,
            PyLogicKind.CLASS: None,
            PyLogicKind.BRANCH: None,
            PyLogicKind.LOOP: None,
            PyLogicKind.TRY: None,
            PyLogicKind.WITH: None,
            PyLogicKind.IMPORT: ImportBuilder.build,
            PyLogicKind.DELETE: None,
            PyLogicKind.RETURN: None,
            PyLogicKind.PASS: None,
            PyLogicKind.COMMENT: None,
            PyLogicKind.STATEMENT: None,
        }

        out: list[PyIRNode] = []
        for node in nodes:
            func = dispatch.get(node.kind)
            if func is None:
                raise ValueError(
                    f"PyLogicKind {node.kind} has no handler in PyIRBuilder"
                )

            result_nodes = func(parent_obj, node)
            out.extend(result_nodes)

        return out

    # endregion

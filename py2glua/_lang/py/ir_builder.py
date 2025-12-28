from pathlib import Path

from ..parse import (
    PyLogicBlockBuilder,
    PyLogicKind,
    PyLogicNode,
)
from .builders import StatementBuilder
from .ir_dataclass import PyIRContext, PyIRFile, PyIRNode


class PyIRBuilder:
    _DISPATCH = {
        PyLogicKind.FUNCTION: None,
        PyLogicKind.CLASS: None,
        PyLogicKind.BRANCH: None,
        PyLogicKind.LOOP: None,
        PyLogicKind.TRY: None,  # Пока не делать в v0.0.1
        PyLogicKind.WITH: None,
        PyLogicKind.IMPORT: None,
        PyLogicKind.DELETE: None,  # Пока не делать в v0.0.1
        PyLogicKind.RETURN: None,
        PyLogicKind.PASS: None,
        PyLogicKind.COMMENT: None,
        PyLogicKind.STATEMENT: StatementBuilder.build,
    }

    @classmethod
    def build_file(cls, source: str, path_to_file: Path | None = None) -> PyIRFile:
        logic_blocks = PyLogicBlockBuilder.build(source)

        context = PyIRContext(
            parent_context=None,
            meta={
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
        out: list[PyIRNode] = []
        for node in nodes:
            func = cls._DISPATCH.get(node.kind)
            if func is None:
                raise ValueError(
                    f"PyLogicKind {node.kind} has no handler in PyIRBuilder"
                )

            result_nodes = func(parent_obj, node)
            out.extend(result_nodes)

        return out

    # endregion

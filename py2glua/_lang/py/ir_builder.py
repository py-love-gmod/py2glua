from pathlib import Path

from ..parse import PyLogicBlockBuilder, PyLogicKind, PyLogicNode
from .builders import (
    BranchBuilder,
    CommentBuilder,
    ImportBuilder,
    LoopBuilder,
    PassBuilder,
    ReturnBuilder,
    StatementBuilder,
    WithBuilder,
)
from .ir_dataclass import PyIRFile, PyIRNode


class PyIRBuilder:
    _DISPATCH = {
        PyLogicKind.FUNCTION: None,
        PyLogicKind.CLASS: None,
        PyLogicKind.BRANCH: BranchBuilder.build,
        PyLogicKind.LOOP: LoopBuilder.build,
        PyLogicKind.TRY: None,  # Пока не делать в v0.0.1
        PyLogicKind.WITH: WithBuilder.build,
        PyLogicKind.IMPORT: ImportBuilder.build,
        PyLogicKind.DELETE: None,  # Пока не делать в v0.0.1
        PyLogicKind.RETURN: ReturnBuilder.build,
        PyLogicKind.PASS: PassBuilder.build,
        PyLogicKind.COMMENT: CommentBuilder.build,
        PyLogicKind.STATEMENT: StatementBuilder.build,
    }

    @classmethod
    def build_file(cls, source: str, path_to_file: Path | None = None) -> PyIRFile:
        logic_blocks = PyLogicBlockBuilder.build(source)

        py_ir_file = PyIRFile(
            line=None,
            offset=None,
            path=path_to_file,
        )

        py_ir_file.body = cls._build_ir_block(logic_blocks)
        return py_ir_file

    # region Core block dispatcher
    @classmethod
    def _build_ir_block(
        cls,
        nodes: list[PyLogicNode],
    ) -> list[PyIRNode]:
        out: list[PyIRNode] = []
        for node in nodes:
            func = cls._DISPATCH.get(node.kind)
            if func is None:
                raise ValueError(
                    f"PyLogicKind {node.kind} has no handler in PyIRBuilder"
                )

            result_nodes = func(node)
            out.extend(result_nodes)

        return out

    # endregion

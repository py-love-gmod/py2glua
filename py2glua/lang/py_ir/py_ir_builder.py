from pathlib import Path

from ..py_logic_block_builder import (
    PyLogicBlockBuilder,
    PyLogicKind,
    PyLogicNode,
)
from .import_analyzer import ImportAnalyzer
from .py_ir_dataclass import PyIRFile, PyIRNode


class PyIRBuilder:
    @classmethod
    def build(cls, source: str, path_to_file: Path | None = None) -> PyIRFile:
        logic_blocks = PyLogicBlockBuilder.build(source)
        py_ir_file = PyIRFile(
            line=None,
            offset=None,
            path=path_to_file,
            meta={
                "imports": [],
                "aliases": {},
                "name_to_id": {},
                "id_to_name": {},
                "globals": {},
            },
        )
        py_ir_file.body = cls._build_ir_block(logic_blocks, py_ir_file)
        return py_ir_file

    @classmethod
    def _build_ir_block(
        cls,
        nodes: list[PyLogicNode],
        file_obj: PyIRFile,
    ) -> list[PyIRNode]:
        dispatch = {
            PyLogicKind.FUNCTION: None,
            PyLogicKind.CLASS: None,
            PyLogicKind.BRANCH: None,
            PyLogicKind.LOOP: None,
            PyLogicKind.TRY: None,
            PyLogicKind.WITH: None,
            PyLogicKind.IMPORT: lambda f, n, i: ImportAnalyzer.build(f, n, i),
            PyLogicKind.DELETE: None,
            PyLogicKind.RETURN: None,
            PyLogicKind.PASS: None,
            PyLogicKind.COMMENT: None,
            PyLogicKind.STATEMENT: None,
        }
        out: list[PyIRNode] = []
        i = 0
        n = len(nodes)
        while i < n:
            node = nodes[i]
            func = dispatch.get(node.kind)
            if func is None:
                raise ValueError(
                    f"PublicLogicKind type {node.kind} no func to build ir"
                )

            offset, result_nodes = func(file_obj, nodes, i)
            out.extend(result_nodes)
            i += offset

        return out

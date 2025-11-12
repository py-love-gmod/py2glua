from pathlib import Path
from typing import Sequence

from ..py_logic_block_builder import (
    PyLogicBlockBuilder,
    PyLogicKind,
    PyLogicNode,
)
from .import_analyzer import ImportAnalyzer
from .py_ir_dataclass import PyIRDel, PyIRFile, PyIRNode, PyIRReturn
from .statement_compiler import StatementCompiler


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
            PyLogicKind.FUNCTION: None,  # has dataclass: true
            PyLogicKind.CLASS: None,  # has dataclass: true
            PyLogicKind.BRANCH: None,  # has dataclass: false TODO
            PyLogicKind.LOOP: None,  # has dataclass: flase TODO
            PyLogicKind.TRY: None,  # has dataclass: false TODO
            PyLogicKind.WITH: None,  # has dataclass: false TODO
            PyLogicKind.IMPORT: ImportAnalyzer.build,  # has dataclass: true
            PyLogicKind.DELETE: cls._build_ir_delete,  # has dataclass: true
            PyLogicKind.RETURN: cls._build_ir_return,  # has dataclass: true
            PyLogicKind.PASS: None,  # has dataclass: false TODO
            PyLogicKind.COMMENT: None,  # has dataclass: false TODO
            PyLogicKind.STATEMENT: cls._build_ir_statement,  # has dataclass: false TODO
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

    @classmethod
    def _build_ir_statement(
        cls,
        file_obj: PyIRFile,
        node_list: list[PyLogicNode],
        start: int,
    ) -> tuple[int, Sequence[PyIRNode]]:
        origin = node_list[start].origin
        if origin is None:
            raise ValueError

        ir_nodes = StatementCompiler.compile_line(origin.tokens, file_obj)
        return (1, ir_nodes)

    @classmethod
    def _build_ir_return(
        cls,
        file_obj: PyIRFile,
        node_list: list[PyLogicNode],
        start: int,
    ) -> tuple[int, Sequence[PyIRNode]]:
        origin = node_list[start].origin
        if origin is None:
            raise ValueError

        ir = StatementCompiler.compile_expres(origin.tokens[1:], file_obj)
        return (
            1,
            [
                PyIRReturn(
                    line=origin.tokens[0].start[0],
                    offset=origin.tokens[0].start[1],
                    value=ir,
                )
            ],
        )

    @classmethod
    def _build_ir_delete(
        cls,
        file_obj: PyIRFile,
        node_list: list[PyLogicNode],
        start: int,
    ) -> tuple[int, Sequence[PyIRNode]]:
        origin = node_list[start].origin
        if origin is None:
            raise ValueError

        ir = StatementCompiler.compile_expres(origin.tokens[1:], file_obj)
        return (
            1,
            [
                PyIRDel(
                    line=origin.tokens[0].start[0],
                    offset=origin.tokens[0].start[1],
                    value=ir,
                )
            ],
        )

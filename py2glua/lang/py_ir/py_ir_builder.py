from pathlib import Path

from ..py_logic_block_builder import (
    PyLogicBlockBuilder,
    PyLogicKind,
    PyLogicNode,
)
from .import_analyzer import ImportAnalyzer
from .py_ir_dataclass import (
    PyIRContext,
    PyIRDel,
    PyIRFile,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
)
from .statement_compiler import StatementCompiler


class PyIRBuilder:
    @classmethod
    def build(cls, source: str, path_to_file: Path | None = None) -> PyIRFile:
        logic_blocks = PyLogicBlockBuilder.build(source)
        py_ir_file = PyIRFile(
            line=None,
            offset=None,
            path=path_to_file,
            context=PyIRContext(),
        )
        py_ir_file.body = cls._build_ir_block(logic_blocks, py_ir_file)
        return py_ir_file

    @classmethod
    def _build_ir_block(
        cls,
        nodes: list[PyLogicNode],
        parent_obj: PyIRNode,
    ) -> list[PyIRNode]:
        dispatch = {
            PyLogicKind.FUNCTION: None,
            PyLogicKind.CLASS: None,
            PyLogicKind.BRANCH: None,  # dataclass TODO
            PyLogicKind.LOOP: None,  # dataclass TODO
            PyLogicKind.TRY: None,  # dataclass TODO
            PyLogicKind.WITH: None,  # dataclass TODO
            PyLogicKind.IMPORT: ImportAnalyzer.build,
            PyLogicKind.DELETE: cls._build_ir_delete,
            PyLogicKind.RETURN: cls._build_ir_return,
            PyLogicKind.PASS: cls._build_ir_pass,
            PyLogicKind.COMMENT: None,
            PyLogicKind.STATEMENT: cls._build_ir_statement,  # dataclass TODO
        }
        out: list[PyIRNode] = []
        for node in nodes:
            func = dispatch.get(node.kind)
            if func is None:
                raise ValueError(
                    f"PublicLogicKind type {node.kind} no func to build ir"
                )

            result_nodes = func(parent_obj, node)
            out.extend(result_nodes)

        return out

    @classmethod
    def _build_ir_statement(
        cls,
        parant_obj: PyIRFile,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origin
        if origin is None:
            raise ValueError

        ir_nodes = StatementCompiler.compile_line(origin.tokens, parant_obj)
        return ir_nodes

    @classmethod
    def _build_ir_return(
        cls,
        parant_obj: PyIRFile,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origin
        if origin is None:
            raise ValueError

        ir = StatementCompiler.compile_expres(origin.tokens[1:], parant_obj)
        return [
            PyIRReturn(
                line=origin.tokens[0].start[0],
                offset=origin.tokens[0].start[1],
                value=ir,
            )
        ]

    @classmethod
    def _build_ir_delete(
        cls,
        parant_obj: PyIRFile,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origin
        if origin is None:
            raise ValueError

        ir = StatementCompiler.compile_expres(origin.tokens[1:], parant_obj)
        return [
            PyIRDel(
                line=origin.tokens[0].start[0],
                offset=origin.tokens[0].start[1],
                value=ir,
            )
        ]

    @classmethod
    def _build_ir_pass(
        cls,
        parant_obj: PyIRFile,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origin
        if origin is None:
            raise ValueError

        return [
            PyIRPass(
                line=origin.tokens[0].start[0],
                offset=origin.tokens[0].start[1],
            )
        ]

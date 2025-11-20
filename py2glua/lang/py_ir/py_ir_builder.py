from pathlib import Path

from ..py_logic_block_builder import (
    PyLogicBlockBuilder,
    PyLogicKind,
    PyLogicNode,
)
from .import_analyzer import ImportAnalyzer
from .py_ir_dataclass import (
    PyIRComment,
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
    def build_file(cls, source: str, path_to_file: Path | None = None) -> PyIRFile:
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
            PyLogicKind.COMMENT: cls._build_ir_comment,
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
        origin = node.origins[0]
        ir_node = StatementCompiler.compile_expres(origin.tokens, parant_obj)
        return [ir_node]

    @classmethod
    def _build_ir_return(
        cls,
        parant_obj: PyIRFile,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origins[0]
        first_token = origin.tokens[0]
        ir = StatementCompiler.compile_expres(origin.tokens[1:], parant_obj)
        return [
            PyIRReturn(
                first_token.start[0],
                first_token.start[1],
                value=ir,
            )
        ]

    @classmethod
    def _build_ir_delete(
        cls,
        parant_obj: PyIRFile,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origins[0]
        first_token = origin.tokens[0]
        ir = StatementCompiler.compile_expres(origin.tokens[1:], parant_obj)
        return [
            PyIRDel(
                first_token.start[0],
                first_token.start[1],
                value=ir,
            )
        ]

    @classmethod
    def _build_ir_pass(
        cls,
        parant_obj: PyIRFile,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        first_token = node.origins[0].tokens[0]
        return [
            PyIRPass(
                first_token.start[0],
                first_token.start[1],
            )
        ]

    @classmethod
    def _build_ir_comment(
        cls,
        parant_obj: PyIRFile,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        tokens = ""
        first_token = node.origins[0].tokens[0]

        for origin in node.origins:
            for tok in origin.tokens:
                tokens += tok.string.removeprefix("# ")

        return [
            PyIRComment(
                first_token.start[0],
                first_token.start[1],
                tokens,
            )
        ]

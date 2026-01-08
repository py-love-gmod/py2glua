from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from .py_parser import PyParser, RawSyntaxNode, RawSyntaxNodeKind


# region Public
class PyLogicKind(Enum):
    DECORATOR = auto()
    FUNCTION = auto()
    CLASS = auto()
    BRANCH = auto()
    BRANCH_PART = auto()
    LOOP = auto()
    TRY = auto()
    WITH = auto()
    IMPORT = auto()
    DELETE = auto()
    RETURN = auto()
    PASS = auto()
    COMMENT = auto()
    STATEMENT = auto()


@dataclass
class PyLogicNode:
    kind: PyLogicKind
    children: list["PyLogicNode"] = field(default_factory=list)

    # Origin-правило:
    # - почти везде 1 origin
    # - исключение: COMMENT может иметь несколько
    origins: list[RawSyntaxNode] = field(default_factory=list)


# endregion


# region Internal
class _LogicBlockKind(Enum):
    DECORATOR = auto()
    FUNCTION = auto()
    CLASS = auto()
    BRANCHING_CONDITION = auto()
    BRANCH_PART = auto()
    LOOPS = auto()
    TRY_EXCEPT = auto()
    WITH_BLOCK = auto()
    IMPORT = auto()
    DELETE = auto()
    RETURN = auto()
    PASS = auto()
    COMMENT = auto()
    STATEMENT = auto()


@dataclass
class _PyLogicBlock:
    kind: _LogicBlockKind
    body: list["_PyLogicBlock"] = field(default_factory=list)
    origins: list[RawSyntaxNode] = field(default_factory=list)


# endregion


class PyLogicBlockBuilder:
    @classmethod
    def build(cls, source: str) -> list[PyLogicNode]:
        raw_nodes = PyParser.parse(source)
        logic_blocks = cls._build_logic_block(raw_nodes)
        return cls._export_to_public(logic_blocks)

    # region Export
    @classmethod
    def _export_to_public(cls, blocks: list[_PyLogicBlock]) -> list[PyLogicNode]:
        kind_map = {
            _LogicBlockKind.DECORATOR: PyLogicKind.DECORATOR,
            _LogicBlockKind.FUNCTION: PyLogicKind.FUNCTION,
            _LogicBlockKind.CLASS: PyLogicKind.CLASS,
            _LogicBlockKind.BRANCHING_CONDITION: PyLogicKind.BRANCH,
            _LogicBlockKind.BRANCH_PART: PyLogicKind.BRANCH_PART,
            _LogicBlockKind.LOOPS: PyLogicKind.LOOP,
            _LogicBlockKind.TRY_EXCEPT: PyLogicKind.TRY,
            _LogicBlockKind.WITH_BLOCK: PyLogicKind.WITH,
            _LogicBlockKind.IMPORT: PyLogicKind.IMPORT,
            _LogicBlockKind.DELETE: PyLogicKind.DELETE,
            _LogicBlockKind.RETURN: PyLogicKind.RETURN,
            _LogicBlockKind.PASS: PyLogicKind.PASS,
            _LogicBlockKind.COMMENT: PyLogicKind.COMMENT,
            _LogicBlockKind.STATEMENT: PyLogicKind.STATEMENT,
        }

        def to_public(b: _PyLogicBlock) -> PyLogicNode:
            return PyLogicNode(
                kind=kind_map[b.kind],
                children=[to_public(ch) for ch in b.body],
                origins=list(b.origins),
            )

        return [to_public(b) for b in blocks]

    # endregion

    # region Helpers
    @staticmethod
    def _expect_block_after(nodes: list[RawSyntaxNode], hdr_idx: int) -> int:
        j = hdr_idx + 1
        if j >= len(nodes) or nodes[j].kind is not RawSyntaxNodeKind.BLOCK:
            raise SyntaxError(f"Expected BLOCK after {nodes[hdr_idx].kind.name}.")

        block = nodes[j]
        if not getattr(block, "tokens", None):
            raise SyntaxError(f"Empty BLOCK after {nodes[hdr_idx].kind.name}.")

        return j

    @classmethod
    def _consume_header_plus_block(
        cls,
        nodes: list[RawSyntaxNode],
        hdr_idx: int,
    ) -> tuple[int, RawSyntaxNode, RawSyntaxNode]:
        blk_idx = cls._expect_block_after(nodes, hdr_idx)
        offset = (blk_idx + 1) - hdr_idx
        return offset, nodes[hdr_idx], nodes[blk_idx]

    # endregion

    # region Core block builder
    @classmethod
    def _build_logic_block(cls, nodes: list[RawSyntaxNode]) -> list[_PyLogicBlock]:
        dispatch: dict[
            RawSyntaxNodeKind,
            Callable[[list[RawSyntaxNode], int], tuple[int, list[_PyLogicBlock]]],
        ] = {
            RawSyntaxNodeKind.DECORATORS: cls._build_logic_decorator,
            RawSyntaxNodeKind.FUNCTION: cls._build_logic_func_block,
            RawSyntaxNodeKind.CLASS: cls._build_logic_class_block,
            RawSyntaxNodeKind.IF: cls._build_logic_branch_chain,
            RawSyntaxNodeKind.TRY: cls._build_logic_try_chain,
            RawSyntaxNodeKind.WHILE: cls._build_logic_loop_block,
            RawSyntaxNodeKind.FOR: cls._build_logic_loop_block,
            RawSyntaxNodeKind.WITH: cls._build_logic_with_block,
            RawSyntaxNodeKind.IMPORT: cls._build_logic_import_block,
            RawSyntaxNodeKind.OTHER: cls._build_logic_statement,
            RawSyntaxNodeKind.DEL: cls._build_logic_delete,
            RawSyntaxNodeKind.PASS: cls._build_logic_pass,
            RawSyntaxNodeKind.RETURN: cls._build_logic_return,
            RawSyntaxNodeKind.COMMENT: cls._build_logic_comment,
            RawSyntaxNodeKind.DOCSTRING: cls._build_logic_comment,
        }

        illegal_solo = {
            RawSyntaxNodeKind.ELSE,
            RawSyntaxNodeKind.ELIF,
            RawSyntaxNodeKind.EXCEPT,
            RawSyntaxNodeKind.FINALLY,
        }

        out: list[_PyLogicBlock] = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if node.kind in illegal_solo:
                raise SyntaxError(
                    f"Unexpected {node.kind.name} without matching header."
                )

            func = dispatch.get(node.kind)
            if func is None:
                raise ValueError(f"RawNodeKind {node.kind} has no logic builder")

            offset, blocks = func(nodes, i)
            out.extend(blocks)
            i += offset

        return out

    # endregion

    # region Concrete block builders
    @classmethod
    def _build_logic_decorator(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        node = nodes[start]
        return 1, [_PyLogicBlock(_LogicBlockKind.DECORATOR, [], origins=[node])]

    @classmethod
    def _build_logic_statement(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        node = nodes[start]
        return 1, [_PyLogicBlock(_LogicBlockKind.STATEMENT, [], origins=[node])]

    @classmethod
    def _build_logic_func_block(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        off, header, block = cls._consume_header_plus_block(nodes, start)
        inner = cls._build_logic_block(block.tokens)
        return off, [_PyLogicBlock(_LogicBlockKind.FUNCTION, inner, origins=[header])]

    @classmethod
    def _build_logic_class_block(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        off, header, block = cls._consume_header_plus_block(nodes, start)
        inner = cls._build_logic_block(block.tokens)
        return off, [_PyLogicBlock(_LogicBlockKind.CLASS, inner, origins=[header])]

    @classmethod
    def _build_logic_loop_block(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        off, header, block = cls._consume_header_plus_block(nodes, start)
        inner = cls._build_logic_block(block.tokens)
        return off, [_PyLogicBlock(_LogicBlockKind.LOOPS, inner, origins=[header])]

    @classmethod
    def _build_logic_with_block(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        off, header, block = cls._consume_header_plus_block(nodes, start)
        inner = cls._build_logic_block(block.tokens)
        return off, [_PyLogicBlock(_LogicBlockKind.WITH_BLOCK, inner, origins=[header])]

    @classmethod
    def _build_logic_branch_chain(cls, nodes: list[RawSyntaxNode], start: int):
        # Find chain start (defensive; normally called on IF)
        i = start
        while i > 0 and nodes[i - 1].kind in (
            RawSyntaxNodeKind.IF,
            RawSyntaxNodeKind.ELIF,
            RawSyntaxNodeKind.ELSE,
        ):
            i -= 1

        if nodes[i].kind is not RawSyntaxNodeKind.IF:
            i = start

        parts: list[_PyLogicBlock] = []
        j = i

        def add_part(hdr: RawSyntaxNode, blk: RawSyntaxNode) -> None:
            inner = cls._build_logic_block(blk.tokens)
            parts.append(
                _PyLogicBlock(_LogicBlockKind.BRANCH_PART, inner, origins=[hdr])
            )

        # First 'if' part
        off, header, block = cls._consume_header_plus_block(nodes, j)
        j += off
        add_part(header, block)

        # BRANCH must have an origin (test requires it), but it must not duplicate IF/ELSE.
        # Use the first suite BLOCK as BRANCH origin. BLOCK is real raw node and is ignored by raw-balance test.
        branch_origin_block = block

        # Following 'elif' / 'else' parts
        while j < len(nodes) and nodes[j].kind in (
            RawSyntaxNodeKind.ELIF,
            RawSyntaxNodeKind.ELSE,
        ):
            off2, hdr2, blk2 = cls._consume_header_plus_block(nodes, j)
            j += off2
            add_part(hdr2, blk2)

        return j - i, [
            _PyLogicBlock(
                _LogicBlockKind.BRANCHING_CONDITION,
                body=parts,
                origins=[branch_origin_block],
            )
        ]

    @classmethod
    def _build_logic_try_chain(cls, nodes: list[RawSyntaxNode], start: int):
        i = start
        while i > 0 and nodes[i - 1].kind in (
            RawSyntaxNodeKind.EXCEPT,
            RawSyntaxNodeKind.ELSE,
            RawSyntaxNodeKind.FINALLY,
        ):
            i -= 1

        if nodes[i].kind is not RawSyntaxNodeKind.TRY:
            i = start

        j = i
        parts: list[_PyLogicBlock] = []
        headers: list[RawSyntaxNode] = []

        off, header, block = cls._consume_header_plus_block(nodes, j)
        j += off
        headers.append(header)
        parts.extend(cls._build_logic_block(block.tokens))

        while j < len(nodes) and nodes[j].kind is RawSyntaxNodeKind.EXCEPT:
            off2, hdr2, blk2 = cls._consume_header_plus_block(nodes, j)
            j += off2
            headers.append(hdr2)
            parts.extend(cls._build_logic_block(blk2.tokens))

        if j < len(nodes) and nodes[j].kind is RawSyntaxNodeKind.ELSE:
            raise SyntaxError("else after try is not supported in py2glua")

        if j < len(nodes) and nodes[j].kind is RawSyntaxNodeKind.FINALLY:
            off4, hdr4, blk4 = cls._consume_header_plus_block(nodes, j)
            j += off4
            headers.append(hdr4)
            parts.extend(cls._build_logic_block(blk4.tokens))

        return j - i, [
            _PyLogicBlock(
                _LogicBlockKind.TRY_EXCEPT,
                parts,
                origins=headers,
            )
        ]

    @classmethod
    def _build_logic_import_block(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        return 1, [_PyLogicBlock(_LogicBlockKind.IMPORT, [], origins=[nodes[start]])]

    @classmethod
    def _build_logic_delete(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        return 1, [_PyLogicBlock(_LogicBlockKind.DELETE, [], origins=[nodes[start]])]

    @classmethod
    def _build_logic_return(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        return 1, [_PyLogicBlock(_LogicBlockKind.RETURN, [], origins=[nodes[start]])]

    @classmethod
    def _build_logic_pass(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        return 1, [_PyLogicBlock(_LogicBlockKind.PASS, [], origins=[nodes[start]])]

    @classmethod
    def _build_logic_comment(
        cls,
        nodes: list[RawSyntaxNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        i = start
        collected: list[RawSyntaxNode] = []

        while i < len(nodes) and nodes[i].kind in (
            RawSyntaxNodeKind.COMMENT,
            RawSyntaxNodeKind.DOCSTRING,
        ):
            collected.append(nodes[i])
            i += 1

        return i - start, [
            _PyLogicBlock(_LogicBlockKind.COMMENT, [], origins=collected)
        ]

    # endregion

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable

from .parser import Parser, RawNode, RawNodeKind


# region Public
class PublicLogicKind(Enum):
    FUNCTION = auto()
    CLASS = auto()
    BRANCH = auto()
    LOOP = auto()
    TRY = auto()
    WITH = auto()
    IMPORT = auto()
    STATEMENT = auto()


@dataclass
class PublicLogicNode:
    kind: PublicLogicKind
    children: list["PublicLogicNode"] = field(default_factory=list)
    origin: RawNode | None = None


# endregion


# region Internal
class _LogicBlockKind(Enum):
    FUNCTION = auto()
    CLASS = auto()
    BRANCHING_CONDITION = auto()
    LOOPS = auto()
    TRY_EXCEPT = auto()
    WITH_BLOCK = auto()
    IMPORT = auto()
    STATEMENT = auto()


@dataclass
class _PyLogicBlock:
    kind: _LogicBlockKind
    body: list["_PyLogicBlock"] = field(default_factory=list)
    origin: RawNode | None = None


# endregion


class PyLogicBlockBuilder:
    @classmethod
    def build(cls, path_to_file: Path) -> list[PublicLogicNode]:
        source = path_to_file.read_text(encoding="utf-8-sig")
        raw_nodes = Parser.parse(source)
        logic_blocks = cls._build_logic_block(raw_nodes)
        return cls._export_to_public(logic_blocks)

    # region Export
    @classmethod
    def _export_to_public(cls, blocks: list[_PyLogicBlock]) -> list[PublicLogicNode]:
        kind_map = {
            _LogicBlockKind.FUNCTION: PublicLogicKind.FUNCTION,
            _LogicBlockKind.CLASS: PublicLogicKind.CLASS,
            _LogicBlockKind.BRANCHING_CONDITION: PublicLogicKind.BRANCH,
            _LogicBlockKind.LOOPS: PublicLogicKind.LOOP,
            _LogicBlockKind.TRY_EXCEPT: PublicLogicKind.TRY,
            _LogicBlockKind.WITH_BLOCK: PublicLogicKind.WITH,
            _LogicBlockKind.IMPORT: PublicLogicKind.IMPORT,
            _LogicBlockKind.STATEMENT: PublicLogicKind.STATEMENT,
        }

        def to_public(b: _PyLogicBlock) -> PublicLogicNode:
            return PublicLogicNode(
                kind=kind_map[b.kind],
                children=[to_public(ch) for ch in b.body],
                origin=b.origin,
            )

        return [to_public(b) for b in blocks]

    # endregion

    # region Helpers
    @staticmethod
    def _expect_block_after(nodes: list[RawNode], hdr_idx: int) -> int:
        j = hdr_idx + 1
        if j >= len(nodes) or nodes[j].kind is not RawNodeKind.BLOCK:
            raise SyntaxError(f"Expected BLOCK after {nodes[hdr_idx].kind.name}.")

        block = nodes[j]
        if not getattr(block, "tokens", None):
            raise SyntaxError(f"Empty BLOCK after {nodes[hdr_idx].kind.name}.")

        return j

    @classmethod
    def _consume_header_plus_block(
        cls,
        nodes: list[RawNode],
        hdr_idx: int,
    ) -> tuple[int, RawNode, RawNode]:
        blk_idx = cls._expect_block_after(nodes, hdr_idx)
        offset = (blk_idx + 1) - hdr_idx
        return offset, nodes[hdr_idx], nodes[blk_idx]

    # endregion

    # region Core block builder
    @classmethod
    def _build_logic_block(cls, nodes: list[RawNode]) -> list[_PyLogicBlock]:
        dispatch: dict[
            RawNodeKind, Callable[[list[RawNode], int], tuple[int, list[_PyLogicBlock]]]
        ] = {
            RawNodeKind.DECORATORS: cls._build_logic_maybe_decorated,
            RawNodeKind.FUNCTION: cls._build_logic_func_block,
            RawNodeKind.CLASS: cls._build_logic_class_block,
            RawNodeKind.IF: cls._build_logic_branch_chain,
            RawNodeKind.TRY: cls._build_logic_try_chain,
            RawNodeKind.WHILE: cls._build_logic_loop_block,
            RawNodeKind.FOR: cls._build_logic_loop_block,
            RawNodeKind.WITH: cls._build_logic_with_block,
            RawNodeKind.IMPORT: cls._build_logic_import_block,
            RawNodeKind.OTHER: cls._build_logic_statement,
        }

        illegal_solo = {
            RawNodeKind.ELSE,
            RawNodeKind.ELIF,
            RawNodeKind.EXCEPT,
            RawNodeKind.FINALLY,
        }

        out: list[_PyLogicBlock] = []
        i = 0
        n = len(nodes)

        while i < n:
            node = nodes[i]

            if node.kind in illegal_solo:
                raise SyntaxError(
                    f"Unexpected {node.kind.name} without matching header."
                )

            func = dispatch.get(node.kind)
            if func is None:
                i += 1
                continue

            offset, blocks = func(nodes, i)
            out.extend(blocks)
            i += offset

        return out

    # endregion

    # region Concrete block builders
    @classmethod
    def _build_logic_statement(
        cls,
        nodes: list[RawNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        node = nodes[start]
        return 1, [_PyLogicBlock(_LogicBlockKind.STATEMENT, [], origin=node)]

    @classmethod
    def _build_logic_maybe_decorated(
        cls,
        nodes: list[RawNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        i = start
        n = len(nodes)

        while i < n and nodes[i].kind is RawNodeKind.DECORATORS:
            i += 1

        if i >= n:
            raise SyntaxError(
                "Decorator sequence must be followed by a function or class."
            )

        header_kind = nodes[i].kind
        if header_kind not in (RawNodeKind.FUNCTION, RawNodeKind.CLASS):
            raise SyntaxError(f"Decorators cannot be applied to {header_kind.name}.")

        blk_idx = cls._expect_block_after(nodes, i)
        header = nodes[i]
        block = nodes[blk_idx]

        inner = cls._build_logic_block(block.tokens)
        kind = (
            _LogicBlockKind.FUNCTION
            if header_kind is RawNodeKind.FUNCTION
            else _LogicBlockKind.CLASS
        )
        return (blk_idx + 1) - start, [_PyLogicBlock(kind, inner, origin=header)]

    @classmethod
    def _build_logic_func_block(
        cls,
        nodes: list[RawNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        off, header, block = cls._consume_header_plus_block(nodes, start)
        inner = cls._build_logic_block(block.tokens)
        return off, [_PyLogicBlock(_LogicBlockKind.FUNCTION, inner, origin=header)]

    @classmethod
    def _build_logic_class_block(
        cls,
        nodes: list[RawNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        off, header, block = cls._consume_header_plus_block(nodes, start)
        inner = cls._build_logic_block(block.tokens)
        return off, [_PyLogicBlock(_LogicBlockKind.CLASS, inner, origin=header)]

    @classmethod
    def _build_logic_loop_block(
        cls,
        nodes: list[RawNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        off, header, block = cls._consume_header_plus_block(nodes, start)
        inner = cls._build_logic_block(block.tokens)
        return off, [_PyLogicBlock(_LogicBlockKind.LOOPS, inner, origin=header)]

    @classmethod
    def _build_logic_with_block(
        cls,
        nodes: list[RawNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        off, header, block = cls._consume_header_plus_block(nodes, start)
        inner = cls._build_logic_block(block.tokens)
        return off, [_PyLogicBlock(_LogicBlockKind.WITH_BLOCK, inner, origin=header)]

    @classmethod
    def _build_logic_branch_chain(cls, nodes: list[RawNode], start: int):
        i = start
        while i > 0 and nodes[i - 1].kind in (
            RawNodeKind.IF,
            RawNodeKind.ELIF,
            RawNodeKind.ELSE,
        ):
            i -= 1

        if nodes[i].kind is not RawNodeKind.IF:
            i = start

        parts_children: list[_PyLogicBlock] = []
        j = i
        n = len(nodes)

        # if
        off, header, block = cls._consume_header_plus_block(nodes, j)
        j += off
        origin_header = header
        parts_children.extend(cls._build_logic_block(block.tokens))

        # elif / else
        while j < n and nodes[j].kind in (RawNodeKind.ELIF, RawNodeKind.ELSE):
            off2, _, block2 = cls._consume_header_plus_block(nodes, j)
            j += off2
            parts_children.extend(cls._build_logic_block(block2.tokens))

        return j - i, [
            _PyLogicBlock(
                _LogicBlockKind.BRANCHING_CONDITION,
                parts_children,
                origin=origin_header,
            )
        ]

    @classmethod
    def _build_logic_try_chain(cls, nodes: list[RawNode], start: int):
        i = start
        while i > 0 and nodes[i - 1].kind in (
            RawNodeKind.EXCEPT,
            RawNodeKind.ELSE,
            RawNodeKind.FINALLY,
        ):
            i -= 1

        if nodes[i].kind is not RawNodeKind.TRY:
            i = start

        j = i
        n = len(nodes)
        parts_children: list[_PyLogicBlock] = []

        # TRY
        off, header, block = cls._consume_header_plus_block(nodes, j)
        j += off
        origin_header = header
        parts_children.extend(cls._build_logic_block(block.tokens))

        # EXCEPT*
        while j < n and nodes[j].kind is RawNodeKind.EXCEPT:
            off2, _, blk2 = cls._consume_header_plus_block(nodes, j)
            j += off2
            parts_children.extend(cls._build_logic_block(blk2.tokens))

        # ELSE?
        if j < n and nodes[j].kind is RawNodeKind.ELSE:
            off3, _, blk3 = cls._consume_header_plus_block(nodes, j)
            j += off3
            parts_children.extend(cls._build_logic_block(blk3.tokens))

        # FINALLY?
        if j < n and nodes[j].kind is RawNodeKind.FINALLY:
            off4, _, blk4 = cls._consume_header_plus_block(nodes, j)
            j += off4
            parts_children.extend(cls._build_logic_block(blk4.tokens))

        return j - i, [
            _PyLogicBlock(
                _LogicBlockKind.TRY_EXCEPT, parts_children, origin=origin_header
            )
        ]

    @classmethod
    def _build_logic_import_block(
        cls,
        nodes: list[RawNode],
        start: int,
    ) -> tuple[int, list[_PyLogicBlock]]:
        node = nodes[start]
        return 1, [_PyLogicBlock(_LogicBlockKind.IMPORT, [], origin=node)]


    # endregion

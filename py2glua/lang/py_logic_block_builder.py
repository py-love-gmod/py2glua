from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List

from .parser import Parser, RawNode, RawNodeKind


# region Public
class PublicLogicKind(Enum):
    FUNCTION = auto()
    CLASS = auto()
    BRANCH = auto()
    LOOP = auto()
    TRY = auto()
    WITH = auto()
    STATEMENT = auto()


@dataclass
class PublicLogicNode:
    kind: PublicLogicKind
    children: List["PublicLogicNode"] = field(default_factory=list)


# endregion


# region Internal
class _LogicBlockKind(Enum):
    FUNCTION = auto()
    CLASS = auto()
    BRANCHING_CONDITION = auto()
    LOOPS = auto()
    TRY_EXCEPT = auto()
    WITH_BLOCK = auto()


@dataclass
class _PyLogicBlock:
    kind: _LogicBlockKind
    body: list["RawNode | _PyLogicBlock"] = field(default_factory=list)


# endregion


class PyLogicBlockBuilder:
    @classmethod
    def build(cls, path_to_file: Path) -> List[PublicLogicNode]:
        blocks = cls._build(path_to_file)
        return cls._export_to_public(blocks)

    # region Build
    @classmethod
    def _export_to_public(cls, blocks: List[_PyLogicBlock]) -> List[PublicLogicNode]:
        def map_kind(k: _LogicBlockKind) -> PublicLogicKind:
            return {
                _LogicBlockKind.FUNCTION: PublicLogicKind.FUNCTION,
                _LogicBlockKind.CLASS: PublicLogicKind.CLASS,
                _LogicBlockKind.BRANCHING_CONDITION: PublicLogicKind.BRANCH,
                _LogicBlockKind.LOOPS: PublicLogicKind.LOOP,
                _LogicBlockKind.TRY_EXCEPT: PublicLogicKind.TRY,
                _LogicBlockKind.WITH_BLOCK: PublicLogicKind.WITH,
            }.get(k, PublicLogicKind.STATEMENT)

        def to_public(node) -> PublicLogicNode:
            if isinstance(node, _PyLogicBlock):
                children = [to_public(ch) for ch in node.body]
                return PublicLogicNode(kind=map_kind(node.kind), children=children)

            if isinstance(node, RawNode):
                if node.kind is RawNodeKind.BLOCK:
                    return PublicLogicNode(
                        kind=PublicLogicKind.WITH,
                        children=[to_public(ch) for ch in getattr(node, "tokens", [])],
                    )

                return PublicLogicNode(kind=PublicLogicKind.STATEMENT, children=[])

            raise TypeError(f"Unexpected node type during export: {type(node)}")

        return [to_public(b) for b in blocks]

    @classmethod
    def _build(cls, path_to_file: Path) -> list[_PyLogicBlock]:
        source = path_to_file.read_text(encoding="utf-8-sig")
        raw_nodes = Parser.parse(source)
        logic_blocks = cls._build_logic_block(raw_nodes)
        return cls._normalize_logic_blocks(logic_blocks)

    @classmethod
    def _normalize_logic_blocks(
        cls,
        logic_blocks: list[_PyLogicBlock],
    ) -> list[_PyLogicBlock]:
        def convert_rawnode_to_logic(node: RawNode) -> _PyLogicBlock:
            if node.kind is RawNodeKind.BLOCK:
                inner_items = []
                for r in node.tokens:
                    if isinstance(r, _PyLogicBlock):
                        inner_items.append(r)

                    elif isinstance(r, RawNode):
                        inner_items.append(convert_rawnode_to_logic(r))

                    else:
                        raise TypeError(f"Unexpected token inside BLOCK: {type(r)}")

                return _PyLogicBlock(_LogicBlockKind.WITH_BLOCK, inner_items)

            return _PyLogicBlock(_LogicBlockKind.BRANCHING_CONDITION, [])

        def recurse(block: _PyLogicBlock) -> _PyLogicBlock:
            new_body: list[_PyLogicBlock] = []
            for elem in block.body:
                if isinstance(elem, _PyLogicBlock):
                    new_body.append(recurse(elem))

                elif isinstance(elem, RawNode):
                    new_body.append(convert_rawnode_to_logic(elem))

                else:
                    raise TypeError(f"Unexpected element type in body: {type(elem)}")

            block.body = new_body  # pyright: ignore[reportAttributeAccessIssue]
            return block

        return [recurse(b) for b in logic_blocks]

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
        cls, nodes: list[RawNode], hdr_idx: int, parts: list[RawNode]
    ) -> int:
        blk_idx = cls._expect_block_after(nodes, hdr_idx)
        parts.extend((nodes[hdr_idx], nodes[blk_idx]))
        return blk_idx + 1

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

        for block in out:
            for elem in list(block.body):
                if isinstance(elem, RawNode) and elem.kind is RawNodeKind.BLOCK:
                    inner_logic = cls._build_logic_block(elem.tokens)
                    elem.tokens = inner_logic

        return out

    # endregion

    # region Concrete block builders
    @classmethod
    def _build_logic_maybe_decorated(
        cls, nodes: list[RawNode], start: int
    ) -> tuple[int, list[_PyLogicBlock]]:
        i = start
        n = len(nodes)

        while i < n and nodes[i].kind is RawNodeKind.DECORATORS:
            i += 1

        if i >= n:
            raise SyntaxError(
                "Decorator sequence must be followed by a function or class."
            )

        header = nodes[i].kind
        if header not in (RawNodeKind.FUNCTION, RawNodeKind.CLASS):
            raise SyntaxError(f"Decorators cannot be applied to {header.name}.")

        if i == start:
            raise SyntaxError("Expected at least one decorator before function/class.")

        blk_idx = cls._expect_block_after(nodes, i)
        end = blk_idx + 1
        kind = (
            _LogicBlockKind.FUNCTION
            if header is RawNodeKind.FUNCTION
            else _LogicBlockKind.CLASS
        )
        return end - start, [_PyLogicBlock(kind, nodes[start:end])]  # pyright: ignore[reportArgumentType]

    @classmethod
    def _build_logic_func_block(cls, nodes: list[RawNode], start: int):
        blk_idx = cls._expect_block_after(nodes, start)
        return blk_idx + 1 - start, [
            _PyLogicBlock(_LogicBlockKind.FUNCTION, nodes[start : blk_idx + 1])  # pyright: ignore[reportArgumentType]
        ]

    @classmethod
    def _build_logic_class_block(cls, nodes: list[RawNode], start: int):
        blk_idx = cls._expect_block_after(nodes, start)
        return blk_idx + 1 - start, [
            _PyLogicBlock(_LogicBlockKind.CLASS, nodes[start : blk_idx + 1])  # pyright: ignore[reportArgumentType]
        ]

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

        parts: list[RawNode] = []
        j = cls._consume_header_plus_block(nodes, i, parts)
        n = len(nodes)

        while j < n and nodes[j].kind in (RawNodeKind.ELIF, RawNodeKind.ELSE):
            j = cls._consume_header_plus_block(nodes, j, parts)

        if not parts:
            raise SyntaxError("Malformed branching chain.")

        return j - i, [_PyLogicBlock(_LogicBlockKind.BRANCHING_CONDITION, parts)]  # pyright: ignore[reportArgumentType]

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

        parts: list[RawNode] = []
        j = cls._consume_header_plus_block(nodes, i, parts)
        n = len(nodes)

        saw_except = False
        saw_else = False
        saw_finally = False

        while j < n and nodes[j].kind is RawNodeKind.EXCEPT:
            saw_except = True
            j = cls._consume_header_plus_block(nodes, j, parts)

        if j < n and nodes[j].kind is RawNodeKind.ELSE:
            saw_else = True
            j = cls._consume_header_plus_block(nodes, j, parts)

        if j < n and nodes[j].kind is RawNodeKind.FINALLY:
            saw_finally = True
            j = cls._consume_header_plus_block(nodes, j, parts)

        if not (saw_except or saw_finally):
            raise SyntaxError(
                "TRY chain must contain at least one EXCEPT or FINALLY part."
            )

        if saw_else and not saw_except:
            raise SyntaxError("ELSE in TRY chain without EXCEPT is invalid.")

        return j - i, [_PyLogicBlock(_LogicBlockKind.TRY_EXCEPT, parts)]  # pyright: ignore[reportArgumentType]

    @classmethod
    def _build_logic_loop_block(cls, nodes: list[RawNode], start: int):
        blk_idx = cls._expect_block_after(nodes, start)
        return blk_idx + 1 - start, [
            _PyLogicBlock(_LogicBlockKind.LOOPS, nodes[start : blk_idx + 1])  # pyright: ignore[reportArgumentType]
        ]

    @classmethod
    def _build_logic_with_block(cls, nodes: list[RawNode], start: int):
        blk_idx = cls._expect_block_after(nodes, start)
        return blk_idx + 1 - start, [
            _PyLogicBlock(_LogicBlockKind.WITH_BLOCK, nodes[start : blk_idx + 1])  # pyright: ignore[reportArgumentType]
        ]

    # endregion

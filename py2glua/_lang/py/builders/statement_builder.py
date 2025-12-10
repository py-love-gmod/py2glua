import tokenize
from ast import literal_eval
from typing import List, Optional, Sequence

from ...parse import PyLogicNode
from ..ir_dataclass import (
    PyAugAssignType,
    PyBinOPType,
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBinOP,
    PyIRCall,
    PyIRConstant,
    PyIRDict,
    PyIRDictItem,
    PyIRList,
    PyIRNode,
    PyIRSet,
    PyIRSubscript,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarUse,
    PyUnaryOPType,
)


# region Internal token stream helper
class _Stream:
    def __init__(self, tokens: List[tokenize.TokenInfo]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self, value: int = 0) -> Optional[tokenize.TokenInfo]:
        if self.index + value >= len(self.tokens):
            return None

        return self.tokens[self.index + value]

    def advance(self) -> Optional[tokenize.TokenInfo]:
        tok = self.peek()
        if tok is not None:
            self.index += 1
        return tok

    def expect_op(self, value: str) -> tokenize.TokenInfo:
        tok = self.advance()
        if tok is None or tok.type != tokenize.OP or tok.string != value:
            raise SyntaxError(f"Expected {value!r}, got {tok!r}")
        return tok

    def eof(self) -> bool:
        return self.index >= len(self.tokens)


# endregion


class StatementBuilder:
    # region helpers: assignment detection
    @staticmethod
    def _find_assign_ops(tokens: List[tokenize.TokenInfo]):
        AUG = {
            "+=",
            "-=",
            "*=",
            "/=",
            "//=",
            "%=",
            "**=",
            "&=",
            "|=",
            "^=",
            "<<=",
            ">>=",
        }

        depth_paren = depth_brack = depth_brace = 0
        eq_indices: list[int] = []
        aug_index: Optional[int] = None
        aug_op: Optional[str] = None

        for i, t in enumerate(tokens):
            if t.type == tokenize.OP:
                if t.string == "(":
                    depth_paren += 1

                elif t.string == ")":
                    depth_paren -= 1

                elif t.string == "[":
                    depth_brack += 1

                elif t.string == "]":
                    depth_brack -= 1

                elif t.string == "{":
                    depth_brace += 1

                elif t.string == "}":
                    depth_brace -= 1

                elif depth_paren == depth_brack == depth_brace == 0:
                    if t.string == "=":
                        eq_indices.append(i)

                    elif t.string in AUG:
                        if aug_index is not None:
                            raise SyntaxError("Multiple augmented assignments")

                        aug_index = i
                        aug_op = t.string

        if aug_index is not None:
            return "augassign", aug_index, aug_op

        if eq_indices:
            return "assign", eq_indices

        return None

    @staticmethod
    def _split_top_level_commas(
        tokens: List[tokenize.TokenInfo],
    ) -> List[List[tokenize.TokenInfo]]:
        out = []
        acc = []

        depth = 0
        for t in tokens:
            if t.type == tokenize.OP:
                if t.string in "([{":
                    depth += 1
                elif t.string in ")]}":
                    depth -= 1

            if t.string == "," and depth == 0:
                out.append(acc)
                acc = []

            else:
                acc.append(t)

        if acc:
            out.append(acc)

        return out

    # endregion

    # region Public entry
    @staticmethod
    def build(
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> Sequence[PyIRNode]:
        if not node.origins:
            raise ValueError("PyLogicNode.STATEMENT has no origins")

        raw = node.origins[0]
        tokens: List[tokenize.TokenInfo] = [
            t
            for t in raw.tokens
            if isinstance(t, tokenize.TokenInfo)
            and t.type
            not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
        ]

        if not tokens:
            return []

        StatementBuilder._forbid_slice(tokens)
        StatementBuilder._forbid_comprehension(tokens)

        assign_info = StatementBuilder._find_assign_ops(tokens)

        # AugAssign
        if assign_info and assign_info[0] == "augassign":
            _, idx, op_str = assign_info
            assert op_str is not None

            return [StatementBuilder._build_augassign(tokens, idx, op_str)]

        # 2Assign
        if assign_info and assign_info[0] == "assign":
            _, eq_indices = assign_info
            return StatementBuilder._build_assign(tokens, eq_indices)

        # Expression-statement
        stream = _Stream(tokens)
        expr = StatementBuilder._parse_expression(stream)
        if not stream.eof():
            raise SyntaxError("Unexpected tokens at end of expression")

        return [expr]

    # endregion

    # region Forbidden constructs
    @staticmethod
    def _forbid_slice(tokens: List[tokenize.TokenInfo]) -> None:
        stack = []

        for i, t in enumerate(tokens):
            if t.type == tokenize.OP:
                if t.string in "([{":
                    stack.append(t.string)
                elif t.string in ")]}":
                    if stack:
                        stack.pop()

                elif t.string == ":" and stack and stack[-1] == "[":
                    prev = tokens[i - 1] if i > 0 else None
                    nxt = tokens[i + 1] if i + 1 < len(tokens) else None

                    if (
                        prev
                        and nxt
                        and prev.string not in "[,"
                        and nxt.string not in ",]"
                    ):
                        line, col = t.start
                        raise SyntaxError(
                            f"Slicing syntax is not supported in py2glua\n"
                            f"LINE|OFFSET: {line}|{col}"
                        )

    @staticmethod
    def _forbid_comprehension(tokens: List[tokenize.TokenInfo]) -> None:
        stack = []

        for i, t in enumerate(tokens):
            if t.type == tokenize.OP:
                if t.string in "([{":
                    stack.append(t.string)
                elif t.string in ")]}":
                    if stack:
                        stack.pop()

            if t.type == tokenize.NAME and t.string == "for":
                if not stack:
                    continue

                for j in range(i + 1, len(tokens)):
                    tj = tokens[j]
                    if tj.type == tokenize.NAME and tj.string == "in":
                        line, col = t.start
                        raise SyntaxError(
                            f"Comprehensions are not supported in py2glua\n"
                            f"LINE|OFFSET: {line}|{col}"
                        )

                continue

    # endregion

    # region Assignment builders
    @staticmethod
    def _build_augassign(
        tokens: List[tokenize.TokenInfo], op_index: int, op_str: str
    ) -> PyIRAugAssign:
        left_toks = tokens[:op_index]
        right_toks = tokens[op_index + 1 :]

        if not left_toks:
            raise SyntaxError("Missing target for augmented assignment")
        if not right_toks:
            raise SyntaxError("Missing value for augmented assignment")

        stream_left = _Stream(left_toks)
        target = StatementBuilder._parse_postfix(stream_left)
        if not stream_left.eof():
            raise SyntaxError("Invalid target in augmented assignment")

        stream_right = _Stream(right_toks)
        value = StatementBuilder._parse_expression(stream_right)
        if not stream_right.eof():
            raise SyntaxError("Invalid value in augmented assignment")

        op_map = {
            "+=": PyAugAssignType.ADD,
            "-=": PyAugAssignType.SUB,
            "*=": PyAugAssignType.MUL,
            "/=": PyAugAssignType.DIV,
            "//=": PyAugAssignType.FLOORDIV,
            "%=": PyAugAssignType.MOD,
            "**=": PyAugAssignType.POW,
            "&=": PyAugAssignType.BIT_AND,
            "|=": PyAugAssignType.BIT_OR,
            "^=": PyAugAssignType.BIT_XOR,
            "<<=": PyAugAssignType.LSHIFT,
            ">>=": PyAugAssignType.RSHIFT,
        }

        op_enum = op_map.get(op_str)
        if op_enum is None:
            raise SyntaxError(f"Unknown augmented assignment operator: {op_str!r}")

        line = left_toks[0].start[0]
        col = left_toks[0].start[1]

        return PyIRAugAssign(
            line=line,
            offset=col,
            target=target,
            op=op_enum,
            value=value,
        )

    @staticmethod
    def _build_assign(
        tokens: List[tokenize.TokenInfo], eq_indices: List[int]
    ) -> List[PyIRAssign]:
        if len(eq_indices) > 1:
            last_eq = eq_indices[-1]
            rhs_tokens = tokens[last_eq + 1 :]

            stream_r = _Stream(rhs_tokens)
            rhs_expr = StatementBuilder._parse_expression(stream_r)
            if not stream_r.eof():
                raise SyntaxError("Invalid expression on RHS of assignment")

            assigns: List[PyIRAssign] = []

            prev_value = rhs_expr

            splits = []
            start = 0
            for idx in eq_indices:
                splits.append(tokens[start:idx])
                start = idx + 1

            prev_value = rhs_expr

            assigns = []
            for lhs_toks in reversed(splits):
                if not lhs_toks:
                    raise SyntaxError("Missing left-hand side in assignment")

                stream_l = _Stream(lhs_toks)
                lhs_expr = StatementBuilder._parse_postfix(stream_l)
                if not stream_l.eof():
                    raise SyntaxError("Unsupported complex assignment target")

                line = lhs_toks[0].start[0]
                col = lhs_toks[0].start[1]

                assigns.append(
                    PyIRAssign(
                        line=line,
                        offset=col,
                        targets=[lhs_expr],
                        value=prev_value,
                    )
                )

                prev_value = lhs_expr

            assigns.reverse()
            return assigns

        eq = eq_indices[0]
        lhs_tokens = tokens[:eq]
        rhs_tokens = tokens[eq + 1 :]

        if not lhs_tokens:
            raise SyntaxError("Missing LHS in assignment")

        if not rhs_tokens:
            raise SyntaxError("Missing RHS in assignment")

        lhs_parts = StatementBuilder._split_top_level_commas(lhs_tokens)

        stream_r = _Stream(rhs_tokens)
        rhs_expr = StatementBuilder._parse_expression(stream_r)
        if not stream_r.eof():
            raise SyntaxError("Invalid RHS in assignment")

        assigns = []

        if len(lhs_parts) == 1:
            stream_l = _Stream(lhs_parts[0])
            lhs_expr = StatementBuilder._parse_postfix(stream_l)
            if not stream_l.eof():
                raise SyntaxError("Unsupported target expression")

            line = lhs_parts[0][0].start[0]
            col = lhs_parts[0][0].start[1]

            assigns.append(
                PyIRAssign(
                    line=line,
                    offset=col,
                    targets=[lhs_expr],
                    value=rhs_expr,
                )
            )
            return assigns

        targets = []
        for part in lhs_parts:
            stream_l = _Stream(part)
            t_expr = StatementBuilder._parse_postfix(stream_l)
            if not stream_l.eof():
                raise SyntaxError("Invalid target in tuple assignment")
            targets.append(t_expr)

        line = lhs_parts[0][0].start[0]
        col = lhs_parts[0][0].start[1]

        assigns.append(
            PyIRAssign(
                line=line,
                offset=col,
                targets=targets,
                value=rhs_expr,
            )
        )

        return assigns

    # endregion

    # region Expression parser (with precedence)
    @staticmethod
    def _parse_expression(stream: _Stream) -> PyIRNode:
        # Логический or, and, сравнения, битовые, арифметика, унарные, postfix/атом
        return StatementBuilder._parse_or(stream)

    @staticmethod
    def _parse_or(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_and(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.NAME and tok.string == "or":
                stream.advance()
                right = StatementBuilder._parse_and(stream)
                node = PyIRBinOP(
                    line=node.line,
                    offset=node.offset,
                    op=PyBinOPType.OR,
                    left=node,
                    right=right,
                )

            else:
                break

        return node

    @staticmethod
    def _parse_and(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_compare(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.NAME and tok.string == "and":
                stream.advance()
                right = StatementBuilder._parse_compare(stream)
                node = PyIRBinOP(
                    line=node.line,
                    offset=node.offset,
                    op=PyBinOPType.AND,
                    left=node,
                    right=right,
                )

            else:
                break

        return node

    @staticmethod
    def _parse_compare(stream: "_Stream") -> PyIRNode:
        node = StatementBuilder._parse_bit_or(stream)

        while True:
            tok = stream.peek()
            if tok is None:
                break

            op_enum: PyBinOPType | None = None

            # Операторы вида == != < > <= >= (tokenize.OP)
            if tok.type == tokenize.OP and tok.string in {
                "==",
                "!=",
                "<",
                ">",
                "<=",
                ">=",
            }:
                op_tok = stream.advance()
                assert op_tok is not None

                op_enum = StatementBuilder._map_compare_op(op_tok.string)

            # Ключевые операторы in / not in / is / is not
            elif tok.type == tokenize.NAME:
                # a in b
                if tok.string == "in":
                    stream.advance()
                    op_enum = PyBinOPType.IN

                # a is b  /  a is not b
                elif tok.string == "is":
                    start_idx = stream.index
                    stream.advance()
                    nxt = stream.peek()
                    if nxt and nxt.type == tokenize.NAME and nxt.string == "not":
                        stream.advance()
                        op_enum = PyBinOPType.IS_NOT

                    else:
                        op_enum = PyBinOPType.IS

                # a not in b
                elif tok.string == "not":
                    start_idx = stream.index
                    stream.advance()
                    nxt = stream.peek()
                    if nxt and nxt.type == tokenize.NAME and nxt.string == "in":
                        stream.advance()
                        op_enum = PyBinOPType.NOT_IN

                    else:
                        stream.index = start_idx
                        op_enum = None

            if op_enum is None:
                break

            right = StatementBuilder._parse_bit_or(stream)
            node = PyIRBinOP(
                line=node.line,
                offset=node.offset,
                op=op_enum,
                left=node,
                right=right,
            )

        return node

    @staticmethod
    def _map_compare_op(op: str) -> PyBinOPType:
        mapping = {
            "==": PyBinOPType.EQ,
            "!=": PyBinOPType.NE,
            "<": PyBinOPType.LT,
            ">": PyBinOPType.GT,
            "<=": PyBinOPType.LE,
            ">=": PyBinOPType.GE,
        }
        try:
            return mapping[op]

        except KeyError:
            raise SyntaxError(f"Unknown comparison operator: {op!r}")

    @staticmethod
    def _parse_bit_or(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_bit_xor(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string == "|":
                stream.advance()
                right = StatementBuilder._parse_bit_xor(stream)
                node = PyIRBinOP(
                    line=node.line,
                    offset=node.offset,
                    op=PyBinOPType.BIT_OR,
                    left=node,
                    right=right,
                )

            else:
                break

        return node

    @staticmethod
    def _parse_bit_xor(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_bit_and(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string == "^":
                stream.advance()
                right = StatementBuilder._parse_bit_and(stream)
                node = PyIRBinOP(
                    line=node.line,
                    offset=node.offset,
                    op=PyBinOPType.BIT_XOR,
                    left=node,
                    right=right,
                )

            else:
                break

        return node

    @staticmethod
    def _parse_bit_and(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_shift(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string == "&":
                stream.advance()
                right = StatementBuilder._parse_shift(stream)
                node = PyIRBinOP(
                    line=node.line,
                    offset=node.offset,
                    op=PyBinOPType.BIT_AND,
                    left=node,
                    right=right,
                )

            else:
                break

        return node

    @staticmethod
    def _parse_shift(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_add_sub(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string in {"<<", ">>"}:
                op_tok = stream.advance()
                assert op_tok is not None

                right = StatementBuilder._parse_add_sub(stream)
                op = (
                    PyBinOPType.BIT_LSHIFT
                    if op_tok.string == "<<"
                    else PyBinOPType.BIT_RSHIFT
                )
                node = PyIRBinOP(
                    line=node.line,
                    offset=node.offset,
                    op=op,
                    left=node,
                    right=right,
                )

            else:
                break

        return node

    @staticmethod
    def _parse_add_sub(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_mul_div(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string in {"+", "-"}:
                op_tok = stream.advance()
                assert op_tok is not None

                right = StatementBuilder._parse_mul_div(stream)
                op = PyBinOPType.ADD if op_tok.string == "+" else PyBinOPType.SUB
                node = PyIRBinOP(
                    line=node.line,
                    offset=node.offset,
                    op=op,
                    left=node,
                    right=right,
                )

            else:
                break

        return node

    @staticmethod
    def _parse_mul_div(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_unary(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string in {"*", "/", "//", "%"}:
                op_tok = stream.advance()
                assert op_tok is not None

                right = StatementBuilder._parse_unary(stream)

                if op_tok.string == "*":
                    op = PyBinOPType.MUL

                elif op_tok.string == "/":
                    op = PyBinOPType.DIV

                elif op_tok.string == "//":
                    op = PyBinOPType.FLOORDIV

                else:
                    op = PyBinOPType.MOD

                node = PyIRBinOP(
                    line=node.line,
                    offset=node.offset,
                    op=op,
                    left=node,
                    right=right,
                )

            else:
                break

        return node

    @staticmethod
    def _parse_unary(stream: _Stream) -> PyIRNode:
        tok = stream.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input in unary expression")

        # +x, -x, ~x, not x
        if tok.type == tokenize.OP and tok.string in {"+", "-", "~"}:
            op_tok = stream.advance()
            assert op_tok is not None

            operand = StatementBuilder._parse_unary(stream)
            line, col = op_tok.start
            if op_tok.string == "+":
                op = PyUnaryOPType.PLUS

            elif op_tok.string == "-":
                op = PyUnaryOPType.MINUS

            else:
                op = PyUnaryOPType.BIT_INV

            return PyIRUnaryOP(
                line=line,
                offset=col,
                op=op,
                value=operand,
            )

        if tok.type == tokenize.NAME and tok.string == "not":
            op_tok = stream.advance()
            assert op_tok is not None

            operand = StatementBuilder._parse_unary(stream)
            line, col = op_tok.start
            return PyIRUnaryOP(
                line=line,
                offset=col,
                op=PyUnaryOPType.NOT,
                value=operand,
            )

        return StatementBuilder._parse_postfix(stream)

    # endregion

    # region Primary + postfix (.attr, [idx], (args))
    @staticmethod
    def _parse_postfix(stream: _Stream) -> PyIRNode:
        node = StatementBuilder._parse_atom(stream)

        while True:
            tok = stream.peek()
            if tok is None:
                break

            # Attribute: .name
            if tok.type == tokenize.OP and tok.string == ".":
                stream.advance()
                name_tok = stream.advance()
                if name_tok is None or name_tok.type != tokenize.NAME:
                    raise SyntaxError("Expected attribute name after '.'")

                line = node.line
                col = node.offset
                node = PyIRAttribute(
                    line=line,
                    offset=col,
                    value=node,
                    attr=name_tok.string,
                )
                continue

            # Call: (args)
            if tok.type == tokenize.OP and tok.string == "(":
                node = StatementBuilder._parse_call(stream, node)
                continue

            # Subscript: [expr]
            if tok.type == tokenize.OP and tok.string == "[":
                node = StatementBuilder._parse_subscript(stream, node)
                continue

            break

        return node

    @staticmethod
    def _parse_atom(stream: _Stream) -> PyIRNode:
        tok = stream.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input in primary expression")

        # Имя
        if tok.type == tokenize.NAME:
            tok = stream.advance()
            assert tok is not None

            line, col = tok.start
            return PyIRVarUse(
                line=line,
                offset=col,
                name=tok.string,
            )

        # Константа: число или строка
        if tok.type in (tokenize.NUMBER, tokenize.STRING):
            tok = stream.advance()
            assert tok is not None

            line, col = tok.start
            try:
                value = literal_eval(tok.string)

            except Exception:
                value = tok.string

            return PyIRConstant(
                line=line,
                offset=col,
                value=value,
            )

        # ( expr )
        if tok.type == tokenize.OP and tok.string == "(":
            stream.advance()
            node = StatementBuilder._parse_expression(stream)
            stream.expect_op(")")
            return node

        # [a, b, c]
        if tok.type == tokenize.OP and tok.string == "[":
            return StatementBuilder._parse_list_literal(stream)

        # {a, b} или {k: v}
        if tok.type == tokenize.OP and tok.string == "{":
            return StatementBuilder._parse_dict_or_set_literal(stream)

        raise SyntaxError(f"Unexpected token in primary expression: {tok!r}")

    @staticmethod
    def _parse_call(stream: _Stream, func_node: PyIRNode) -> PyIRCall:
        lpar = stream.advance()
        assert lpar and lpar.string == "("

        args_p: list[PyIRNode] = []
        args_kw: dict[str, PyIRNode] = {}

        tok = stream.peek()
        if not (tok and tok.type == tokenize.OP and tok.string == ")"):
            while True:
                t1 = stream.peek()
                t2 = stream.peek(1)

                if (
                    t1
                    and t1.type == tokenize.NAME
                    and t2
                    and t2.type == tokenize.OP
                    and t2.string == "="
                ):
                    key_tok = stream.advance()
                    assert key_tok is not None

                    stream.advance()  # съесть '='

                    val = StatementBuilder._parse_expression(stream)
                    args_kw[key_tok.string] = val

                else:
                    arg = StatementBuilder._parse_expression(stream)
                    args_p.append(arg)

                tok = stream.peek()
                if tok and tok.type == tokenize.OP and tok.string == ",":
                    stream.advance()
                    continue

                break

        stream.expect_op(")")

        line = func_node.line
        col = func_node.offset

        return PyIRCall(
            line=line,
            offset=col,
            func=func_node,
            args_p=args_p,
            args_kw=args_kw,
        )

    @staticmethod
    def _parse_subscript(stream: _Stream, base: PyIRNode) -> PyIRSubscript:
        lbr = stream.advance()
        assert lbr and lbr.string == "["

        index_expr = StatementBuilder._parse_expression(stream)
        stream.expect_op("]")

        line = base.line
        col = base.offset

        return PyIRSubscript(
            line=line,
            offset=col,
            value=base,
            index=index_expr,
        )

    @staticmethod
    def _parse_list_literal(stream: _Stream) -> PyIRList:
        lbr = stream.advance()
        assert lbr and lbr.string == "["
        elements: list[PyIRNode] = []

        tok = stream.peek()
        if tok and not (tok.type == tokenize.OP and tok.string == "]"):
            while True:
                elem = StatementBuilder._parse_expression(stream)
                elements.append(elem)
                tok = stream.peek()
                if tok and tok.type == tokenize.OP and tok.string == ",":
                    stream.advance()
                    continue

                break

        stream.expect_op("]")
        line, col = lbr.start
        return PyIRList(
            line=line,
            offset=col,
            elements=elements,
        )

    @staticmethod
    def _parse_dict_or_set_literal(stream: _Stream) -> PyIRNode:
        lbr = stream.advance()
        assert lbr and lbr.string == "{"

        items: list[PyIRDictItem] = []
        elements: list[PyIRNode] = []

        tok = stream.peek()
        if tok and not (tok.type == tokenize.OP and tok.string == "}"):
            first_expr = StatementBuilder._parse_expression(stream)
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string == ":":
                stream.advance()
                value_expr = StatementBuilder._parse_expression(stream)
                line = first_expr.line
                col = first_expr.offset
                items.append(
                    PyIRDictItem(
                        line=line,
                        offset=col,
                        key=first_expr,
                        value=value_expr,
                    )
                )

                while True:
                    tok = stream.peek()
                    if tok and tok.type == tokenize.OP and tok.string == ",":
                        stream.advance()
                        tok = stream.peek()
                        if tok and not (tok.type == tokenize.OP and tok.string == "}"):
                            k_expr = StatementBuilder._parse_expression(stream)
                            stream.expect_op(":")
                            v_expr = StatementBuilder._parse_expression(stream)
                            line = k_expr.line
                            col = k_expr.offset
                            items.append(
                                PyIRDictItem(
                                    line=line,
                                    offset=col,
                                    key=k_expr,
                                    value=v_expr,
                                )
                            )
                            continue
                    break

                stream.expect_op("}")
                line, col = lbr.start
                return PyIRDict(
                    line=line,
                    offset=col,
                    items=items,
                )

            else:
                # set-like: {a, b, c}
                elements.append(first_expr)
                while True:
                    tok = stream.peek()
                    if tok and tok.type == tokenize.OP and tok.string == ",":
                        stream.advance()
                        tok = stream.peek()
                        if tok and not (tok.type == tokenize.OP and tok.string == "}"):
                            elem = StatementBuilder._parse_expression(stream)
                            elements.append(elem)
                            continue

                    break

        stream.expect_op("}")
        line, col = lbr.start
        return PyIRSet(
            line=line,
            offset=col,
            elements=elements,
        )

    # endregion

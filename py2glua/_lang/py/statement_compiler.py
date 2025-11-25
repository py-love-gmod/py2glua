import ast
import tokenize

from ..etc import TokenStream
from .py_ir_dataclass import (
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


class StatementCompiler:
    @staticmethod
    def compile_expres(
        tokens: list[tokenize.TokenInfo],
        parant_obj: PyIRNode,
    ) -> PyIRNode:
        filtered: list[tokenize.TokenInfo] = [
            t
            for t in tokens
            if t.type
            not in (
                tokenize.ENCODING,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            )
        ]
        stream = TokenStream(filtered)
        parser = _ExprParser(stream, parant_obj)
        node = parser.parse_assignment_or_expr()

        if not stream.eof() and stream.peek().string == ",":  # pyright: ignore[reportOptionalMemberAccess]
            elements: list[PyIRNode] = [node]
            first_line = node.line
            first_offset = node.offset
            while not stream.eof() and stream.peek().string == ",":  # pyright: ignore[reportOptionalMemberAccess]
                stream.advance()
                if stream.eof():
                    break

                elements.append(parser.parse_expression_no_assign())

            node = PyIRTuple(
                line=first_line,
                offset=first_offset,
                elements=elements,
            )

        return node


class _ExprParser:
    def __init__(self, stream: TokenStream, parent: PyIRNode) -> None:
        self.stream = stream
        self.parent = parent

    def peek(self, offset: int = 0) -> tokenize.TokenInfo | None:
        return self.stream.peek(offset)

    def advance(self) -> tokenize.TokenInfo:
        tok = self.stream.advance()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")

        return tok

    def match(self, *values: str) -> tokenize.TokenInfo | None:
        tok = self.peek()
        if tok is None:
            return None

        if tok.string in values:
            self.advance()
            return tok

        return None

    def expect(self, value: str) -> tokenize.TokenInfo:
        tok = self.advance()
        if tok.string != value:
            raise SyntaxError(f"Expected {value!r}, got {tok.string!r}")

        return tok

    def parse_assignment_or_expr(self) -> PyIRNode:
        lhs = self.parse_expression_no_assign()

        tok = self.peek()
        if tok is None:
            return lhs

        # augmented assignment
        aug_map = {
            "+=": PyAugAssignType.ADD,
            "-=": PyAugAssignType.SUB,
            "*=": PyAugAssignType.MUL,
            "/=": PyAugAssignType.DIV,
            "//=": PyAugAssignType.FLOORDIV,
            "%=": PyAugAssignType.MOD,
            "**=": PyAugAssignType.POW,
            "|=": PyAugAssignType.BIT_OR,
            "^=": PyAugAssignType.BIT_XOR,
            "&=": PyAugAssignType.BIT_AND,
            "<<=": PyAugAssignType.LSHIFT,
            ">>=": PyAugAssignType.RSHIFT,
        }

        if tok.string in aug_map:
            op_tok = self.advance()
            value = self.parse_expression_no_assign()
            return PyIRAugAssign(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                target=lhs,
                op=aug_map[op_tok.string],
                value=value,
            )

        # simple assignment: target = value
        if tok.string == "=":
            eq_tok = self.advance()
            value = self.parse_expression_no_assign()
            return PyIRAssign(
                line=eq_tok.start[0],
                offset=eq_tok.start[1],
                targets=[lhs],
                value=value,
            )

        return lhs

    def parse_expression_no_assign(self) -> PyIRNode:
        return self.parse_or()

    def parse_or(self) -> PyIRNode:
        node = self.parse_and()
        while True:
            tok = self.peek()
            if tok is None or tok.string != "or":
                break

            op_tok = self.advance()
            rhs = self.parse_and()
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=PyBinOPType.OR,
                left=node,
                right=rhs,
            )

        return node

    def parse_and(self) -> PyIRNode:
        node = self.parse_not()
        while True:
            tok = self.peek()
            if tok is None or tok.string != "and":
                break

            op_tok = self.advance()
            rhs = self.parse_not()
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=PyBinOPType.AND,
                left=node,
                right=rhs,
            )

        return node

    def parse_not(self) -> PyIRNode:
        tok = self.peek()
        if tok is not None and tok.string == "not":
            nxt = self.peek(1)
            if not (nxt is not None and nxt.string == "in"):
                op_tok = self.advance()
                value = self.parse_not()
                return PyIRUnaryOP(
                    line=op_tok.start[0],
                    offset=op_tok.start[1],
                    op=PyUnaryOPType.NOT,
                    value=value,
                )

        return self.parse_compare()

    def parse_compare(self) -> PyIRNode:
        node = self.parse_bit_or()
        while True:
            tok = self.peek()
            if tok is None:
                break

            op_type: PyBinOPType | None = None
            consume = 1

            if tok.string in ("==", "!=", "<", ">", "<=", ">="):
                mapping = {
                    "==": PyBinOPType.EQ,
                    "!=": PyBinOPType.NE,
                    "<": PyBinOPType.LT,
                    ">": PyBinOPType.GT,
                    "<=": PyBinOPType.LE,
                    ">=": PyBinOPType.GE,
                }
                op_type = mapping[tok.string]

            elif tok.string == "in":
                op_type = PyBinOPType.IN

            elif tok.string == "not":
                nxt = self.peek(1)
                if nxt is not None and nxt.string == "in":
                    op_type = PyBinOPType.NOT_IN
                    consume = 2

                else:
                    break

            elif tok.string == "is":
                nxt = self.peek(1)
                if nxt is not None and nxt.string == "not":
                    op_type = PyBinOPType.IS_NOT
                    consume = 2

                else:
                    op_type = PyBinOPType.IS

            else:
                break

            if op_type is None:
                break

            if consume == 1:
                op_tok = self.advance()
            else:
                op_tok = tok
                self.advance()
                self.advance()

            rhs = self.parse_bit_or()
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=op_type,
                left=node,
                right=rhs,
            )

        return node

    def parse_bit_or(self) -> PyIRNode:
        node = self.parse_bit_xor()
        while True:
            tok = self.peek()
            if tok is None or tok.string != "|":
                break

            op_tok = self.advance()
            rhs = self.parse_bit_xor()
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=PyBinOPType.BIT_OR,
                left=node,
                right=rhs,
            )

        return node

    def parse_bit_xor(self) -> PyIRNode:
        node = self.parse_bit_and()
        while True:
            tok = self.peek()
            if tok is None or tok.string != "^":
                break

            op_tok = self.advance()
            rhs = self.parse_bit_and()
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=PyBinOPType.BIT_XOR,
                left=node,
                right=rhs,
            )

        return node

    def parse_bit_and(self) -> PyIRNode:
        node = self.parse_shift()
        while True:
            tok = self.peek()
            if tok is None or tok.string != "&":
                break

            op_tok = self.advance()
            rhs = self.parse_shift()
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=PyBinOPType.BIT_AND,
                left=node,
                right=rhs,
            )

        return node

    def parse_shift(self) -> PyIRNode:
        node = self.parse_add()
        while True:
            tok = self.peek()
            if tok is None or tok.string not in ("<<", ">>"):
                break

            op_tok = self.advance()
            rhs = self.parse_add()
            op_type = (
                PyBinOPType.BIT_LSHIFT
                if op_tok.string == "<<"
                else PyBinOPType.BIT_RSHIFT
            )
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=op_type,
                left=node,
                right=rhs,
            )

        return node

    def parse_add(self) -> PyIRNode:
        node = self.parse_mul()
        while True:
            tok = self.peek()
            if tok is None or tok.string not in ("+", "-"):
                break

            op_tok = self.advance()
            rhs = self.parse_mul()
            op_type = PyBinOPType.ADD if op_tok.string == "+" else PyBinOPType.SUB
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=op_type,
                left=node,
                right=rhs,
            )

        return node

    def parse_mul(self) -> PyIRNode:
        node = self.parse_power()
        while True:
            tok = self.peek()
            if tok is None or tok.string not in ("*", "/", "//", "%"):
                break

            op_tok = self.advance()
            rhs = self.parse_power()
            mapping = {
                "*": PyBinOPType.MUL,
                "/": PyBinOPType.DIV,
                "//": PyBinOPType.FLOORDIV,
                "%": PyBinOPType.MOD,
            }

            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=mapping[op_tok.string],
                left=node,
                right=rhs,
            )

        return node

    def parse_power(self) -> PyIRNode:
        node = self.parse_unary()
        tok = self.peek()
        if tok is not None and tok.string == "**":
            op_tok = self.advance()
            rhs = self.parse_power()
            node = PyIRBinOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=PyBinOPType.POW,
                left=node,
                right=rhs,
            )
        return node

    def parse_unary(self) -> PyIRNode:
        tok = self.peek()
        if tok is not None and tok.string in ("+", "-", "~"):
            op_tok = self.advance()
            value = self.parse_unary()
            mapping = {
                "+": PyUnaryOPType.PLUS,
                "-": PyUnaryOPType.MINUS,
                "~": PyUnaryOPType.BIT_INV,
            }
            return PyIRUnaryOP(
                line=op_tok.start[0],
                offset=op_tok.start[1],
                op=mapping[op_tok.string],
                value=value,
            )
        return self.parse_primary()

    def parse_primary(self) -> PyIRNode:
        node = self.parse_atom()

        while True:
            tok = self.peek()
            if tok is None:
                break

            if tok.string == ".":
                dot_tok = self.advance()
                name_tok = self.advance()
                if name_tok.type != tokenize.NAME:
                    raise SyntaxError("Expected attribute name after '.'")
                node = PyIRAttribute(
                    line=dot_tok.start[0],
                    offset=dot_tok.start[1],
                    value=node,
                    attr=name_tok.string,
                )
                continue

            if tok.string == "(":
                node = self._parse_call(node)
                continue

            if tok.string == "[":
                lbr = self.advance()
                index_expr = self.parse_expression_no_assign()
                self.expect("]")
                node = PyIRSubscript(
                    line=lbr.start[0],
                    offset=lbr.start[1],
                    value=node,
                    index=index_expr,
                )
                continue

            break

        return node

    def parse_atom(self) -> PyIRNode:
        tok = self.advance()

        # (expr) / (a, b) / ()
        if tok.string == "(":
            if self.peek() is not None and self.peek().string == ")":  # pyright: ignore[reportOptionalMemberAccess]
                self.advance()
                return PyIRTuple(
                    line=tok.start[0],
                    offset=tok.start[1],
                    elements=[],
                )

            first_expr = self.parse_expression_no_assign()
            elems = [first_expr]
            while self.peek() is not None and self.peek().string == ",":  # pyright: ignore[reportOptionalMemberAccess]
                self.advance()
                if self.peek() is not None and self.peek().string == ")":  # pyright: ignore[reportOptionalMemberAccess]
                    break

                elems.append(self.parse_expression_no_assign())

            self.expect(")")
            if len(elems) == 1:
                return first_expr

            return PyIRTuple(
                line=tok.start[0],
                offset=tok.start[1],
                elements=elems,
            )

        # [a, b, c]
        if tok.string == "[":
            elements: list[PyIRNode] = []
            if self.peek() is not None and self.peek().string == "]":  # pyright: ignore[reportOptionalMemberAccess]
                self.advance()
            else:
                while True:
                    elements.append(self.parse_expression_no_assign())
                    if self.peek() is None or self.peek().string != ",":  # pyright: ignore[reportOptionalMemberAccess]
                        break
                    self.advance()
                    if self.peek() is not None and self.peek().string == "]":  # pyright: ignore[reportOptionalMemberAccess]
                        break
                self.expect("]")
            return PyIRList(
                line=tok.start[0],
                offset=tok.start[1],
                elements=elements,
            )

        # { } / {a: b} / {a, b}
        if tok.string == "{":
            if self.peek() is not None and self.peek().string == "}":  # pyright: ignore[reportOptionalMemberAccess]
                self.advance()
                return PyIRDict(
                    line=tok.start[0],
                    offset=tok.start[1],
                    items=[],
                )

            save_pos = self.stream.pos
            self.parse_expression_no_assign()
            nxt = self.peek()
            self.stream.pos = save_pos

            if nxt is not None and nxt.string == ":":
                # dict
                items: list[PyIRDictItem] = []
                while True:
                    key_expr = self.parse_expression_no_assign()
                    self.expect(":")
                    value_expr = self.parse_expression_no_assign()
                    items.append(
                        PyIRDictItem(
                            line=key_expr.line,
                            offset=key_expr.offset,
                            key=key_expr,
                            value=value_expr,
                        )
                    )
                    if self.peek() is None or self.peek().string != ",":  # pyright: ignore[reportOptionalMemberAccess]
                        break

                    self.advance()
                    if self.peek() is not None and self.peek().string == "}":  # pyright: ignore[reportOptionalMemberAccess]
                        break

                self.expect("}")
                return PyIRDict(
                    line=tok.start[0],
                    offset=tok.start[1],
                    items=items,
                )

            else:
                # set
                elements: list[PyIRNode] = []
                while True:
                    elements.append(self.parse_expression_no_assign())
                    if self.peek() is None or self.peek().string != ",":  # pyright: ignore[reportOptionalMemberAccess]
                        break

                    self.advance()
                    if self.peek() is not None and self.peek().string == "}":  # pyright: ignore[reportOptionalMemberAccess]
                        break

                self.expect("}")
                return PyIRSet(
                    line=tok.start[0],
                    offset=tok.start[1],
                    elements=elements,
                )

        if tok.type in (tokenize.NUMBER, tokenize.STRING):
            try:
                value = ast.literal_eval(tok.string)

            except Exception:
                value = tok.string

            return PyIRConstant(
                line=tok.start[0],
                offset=tok.start[1],
                value=value,
            )

        if tok.type == tokenize.NAME:
            if tok.string in ("True", "False", "None"):
                mapping = {
                    "True": True,
                    "False": False,
                    "None": None,
                }
                return PyIRConstant(
                    line=tok.start[0],
                    offset=tok.start[1],
                    value=mapping[tok.string],
                )

            return PyIRVarUse(
                line=tok.start[0],
                offset=tok.start[1],
                name=tok.string,
            )

        raise SyntaxError(f"Unexpected token {tok.string!r} in expression")

    def _parse_call(self, func_expr: PyIRNode) -> PyIRNode:
        lpar = self.expect("(")

        args_p: list[PyIRNode] = []
        args_kw: dict[str, PyIRNode] = {}

        if self.peek() is not None and self.peek().string == ")":  # pyright: ignore[reportOptionalMemberAccess]
            self.advance()

        else:
            while True:
                tok = self.peek()
                # keyword-арг: name=
                if (
                    tok is not None
                    and tok.type == tokenize.NAME
                    and self.stream.peek(1) is not None
                    and self.stream.peek(1).string == "="  # pyright: ignore[reportOptionalMemberAccess]
                ):
                    name_tok = self.advance()
                    self.expect("=")
                    value_expr = self.parse_expression_no_assign()
                    args_kw[name_tok.string] = value_expr
                else:
                    args_p.append(self.parse_expression_no_assign())

                tok = self.peek()
                if tok is None or tok.string != ",":
                    break
                self.advance()
                if self.peek() is not None and self.peek().string == ")":  # pyright: ignore[reportOptionalMemberAccess]
                    break

            self.expect(")")

        call_name = ""
        if isinstance(func_expr, PyIRVarUse):
            call_name = func_expr.name

        return PyIRCall(
            line=lpar.start[0],
            offset=lpar.start[1],
            name=call_name,
            args_p=args_p,
            args_kw=args_kw,
        )

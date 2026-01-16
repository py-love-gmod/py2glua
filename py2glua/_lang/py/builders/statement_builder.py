import tokenize
from typing import List, Sequence

from ...etc import TokenStream
from ...parse import PyLogicNode
from ..ir_dataclass import (
    PyAugAssignType,
    PyBinOPType,
    PyIRAnnotatedAssign,
    PyIRAnnotation,
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBinOP,
    PyIRCall,
    PyIRConstant,
    PyIRDict,
    PyIRDictItem,
    PyIRFString,
    PyIRList,
    PyIRNode,
    PyIRSet,
    PyIRSubscript,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarUse,
    PyUnaryOPType,
)


class StatementBuilder:
    # Helpers: assignment detection / comma splitting
    @staticmethod
    def _find_assign_ops(tokens: List[tokenize.TokenInfo]):
        """
        Находит:
        - augassign:  x += y
        - assign:     x = y, a = b = c
        Только на верхнем уровне (не внутри (), [], {}).
        """
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

        depth_paren = 0
        depth_brack = 0
        depth_brace = 0

        eq_indices: list[int] = []
        aug_index: int | None = None
        aug_op: str | None = None

        for i, t in enumerate(tokens):
            if t.type != tokenize.OP:
                continue

            s = t.string
            if s == "(":
                depth_paren += 1
                continue

            if s == ")":
                depth_paren -= 1
                continue

            if s == "[":
                depth_brack += 1
                continue

            if s == "]":
                depth_brack -= 1
                continue

            if s == "{":
                depth_brace += 1
                continue

            if s == "}":
                depth_brace -= 1
                continue

            if depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
                if s == "=":
                    eq_indices.append(i)

                elif s in AUG:
                    if aug_index is not None:
                        raise SyntaxError(
                            "Нельзя использовать несколько операторов расширенного присваивания"
                        )

                    aug_index = i
                    aug_op = s

        if aug_index is not None:
            return "augassign", aug_index, aug_op

        if eq_indices:
            return "assign", eq_indices

        return None

    @staticmethod
    def _split_top_level_commas(
        tokens: List[tokenize.TokenInfo],
    ) -> List[List[tokenize.TokenInfo]]:
        """
        Делит токены по ',' на верхнем уровне (не внутри (), [], {}).
        """
        out: List[List[tokenize.TokenInfo]] = []
        acc: List[tokenize.TokenInfo] = []

        depth = 0
        for t in tokens:
            if t.type == tokenize.OP:
                if t.string in "([{":
                    depth += 1

                elif t.string in ")]}":
                    depth -= 1

            if t.type == tokenize.OP and t.string == "," and depth == 0:
                out.append(acc)
                acc = []

            else:
                acc.append(t)

        if acc:
            out.append(acc)

        return out

    # Helpers: forbidden constructs
    @staticmethod
    def _forbid_slice(tokens: List[tokenize.TokenInfo]) -> None:
        """
        Любой ':' внутри [] считаем slice и рубим.
        Dict не ломается, т.к. там {}.
        """
        stack: List[str] = []

        for t in tokens:
            if t.type != tokenize.OP:
                continue

            s = t.string
            if s in "([{":
                stack.append(s)
                continue

            if s in ")]}":
                if stack:
                    stack.pop()
                continue

            if s == ":" and stack and stack[-1] == "[":
                line, col = t.start
                raise SyntaxError(
                    "Срезы (slice) не поддерживаются в py2glua\n"
                    f"LINE|OFFSET: {line}|{col}"
                )

    @staticmethod
    def _forbid_comprehension(tokens: List[tokenize.TokenInfo]) -> None:
        """
        Любой 'for ... in ...' внутри ()/[]/{} считаем comprehension и рубим.
        """
        stack: List[str] = []
        n = len(tokens)

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

                for j in range(i + 1, n):
                    tj = tokens[j]
                    if tj.type == tokenize.NAME and tj.string == "in":
                        line, col = t.start
                        raise SyntaxError(
                            "Генераторы списков/словари/множества (comprehension) не поддерживаются в py2glua\n"
                            f"LINE|OFFSET: {line}|{col}"
                        )

    # Helpers: annotation parsing
    @staticmethod
    def _parse_annotation_header(
        tokens: List[tokenize.TokenInfo],
    ) -> tuple[tokenize.TokenInfo, str, int | None] | None:
        """
        Распознаёт:
            NAME : TYPE
            NAME : TYPE = EXPR

        Возвращает:
            (name_token, annotation_text, eq_index_or_None)

        Важно:
        - eq_index — индекс токена '=' на верхнем уровне (не внутри (), [], {}), если он есть.
        """
        if not tokens:
            return None

        if tokens[0].type != tokenize.NAME:
            return None

        if len(tokens) < 2:
            return None

        if tokens[1].type != tokenize.OP or tokens[1].string != ":":
            return None

        depth = 0
        ann_tokens: list[tokenize.TokenInfo] = []
        eq_index: int | None = None

        for i in range(2, len(tokens)):
            t = tokens[i]
            if t.type == tokenize.OP:
                if t.string in "([{":
                    depth += 1

                elif t.string in ")]}":
                    depth -= 1

                elif t.string == "=" and depth == 0:
                    eq_index = i
                    break

            ann_tokens.append(t)

        if not ann_tokens:
            raise SyntaxError("Отсутствует тип аннотации после ':'")

        ann_text = "".join(t.string for t in ann_tokens).strip()
        if not ann_text:
            raise SyntaxError("Тип аннотации пустой")

        return tokens[0], ann_text, eq_index

    # Helper
    @staticmethod
    def _parse_expression_until(
        stream: TokenStream,
        stop_ops: set[str],
    ) -> PyIRNode:
        """
        Парсит выражение, но запрещает выход за стоп-операторы
        (используется для dict key / value).
        """
        start = stream.index
        node = StatementBuilder._parse_or(stream)

        tok = stream.peek()
        if tok and tok.type == tokenize.OP and tok.string in stop_ops:
            return node

        stream.index = start
        raise SyntaxError(f"Unexpected token in expression: {tok!r}")

    # Public entry
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
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

        # 1) Аннотация / annotated assign: NAME : TYPE [= EXPR]
        ann = StatementBuilder._parse_annotation_header(tokens)
        if ann is not None:
            name_tok, ann_text, eq_index = ann
            line, col = name_tok.start

            # NAME : TYPE
            if eq_index is None:
                return [
                    PyIRAnnotation(
                        line=line,
                        offset=col,
                        name=name_tok.string,
                        annotation=ann_text,
                    )
                ]

            # NAME : TYPE = EXPR   (annotated assign)
            if eq_index + 1 >= len(tokens):
                raise SyntaxError(
                    "Отсутствует выражение после '=' в аннотированном присваивании"
                )

            rhs_tokens = tokens[eq_index + 1 :]
            stream_r = TokenStream(rhs_tokens)
            rhs_expr = StatementBuilder._parse_expression(stream_r)
            if not stream_r.eof():
                raise SyntaxError(
                    "Лишние токены в правой части аннотированного присваивания"
                )

            return [
                PyIRAnnotatedAssign(
                    line=line,
                    offset=col,
                    name=name_tok.string,
                    annotation=ann_text,
                    value=rhs_expr,
                )
            ]

        # 2) AugAssign / Assign / Expr
        assign_info = StatementBuilder._find_assign_ops(tokens)

        if assign_info and assign_info[0] == "augassign":
            _, idx, op_str = assign_info
            assert op_str is not None  # unreachable
            return [StatementBuilder._build_augassign(tokens, idx, op_str)]

        if assign_info and assign_info[0] == "assign":
            _, eq_indices = assign_info
            return StatementBuilder._build_assign(tokens, eq_indices)

        # 3) Expression-statement
        stream = TokenStream(tokens)
        expr = StatementBuilder._parse_expression(stream)
        if not stream.eof():
            raise SyntaxError("Лишние токены в конце выражения")

        return [expr]

    # Assignment builders
    @staticmethod
    def _build_augassign(
        tokens: List[tokenize.TokenInfo], op_index: int, op_str: str
    ) -> PyIRAugAssign:
        left_toks = tokens[:op_index]
        right_toks = tokens[op_index + 1 :]

        if not left_toks:
            raise SyntaxError("Отсутствует цель для расширенного присваивания")
        if not right_toks:
            raise SyntaxError("Отсутствует значение для расширенного присваивания")

        stream_left = TokenStream(left_toks)
        target = StatementBuilder._parse_postfix(stream_left)
        if not stream_left.eof():
            raise SyntaxError("Некорректная цель в расширенном присваивании")

        stream_right = TokenStream(right_toks)
        value = StatementBuilder._parse_expression(stream_right)
        if not stream_right.eof():
            raise SyntaxError("Некорректное значение в расширенном присваивании")

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
            raise SyntaxError(
                f"Неизвестный оператор расширенного присваивания: {op_str!r}"
            )

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
        # Чейнинг a = b = c = expr
        if len(eq_indices) > 1:
            last_eq = eq_indices[-1]
            rhs_tokens = tokens[last_eq + 1 :]

            if not rhs_tokens:
                raise SyntaxError("Отсутствует правая часть в цепочке присваиваний")

            stream_r = TokenStream(rhs_tokens)
            rhs_expr = StatementBuilder._parse_expression(stream_r)
            if not stream_r.eof():
                raise SyntaxError("Некорректное выражение в правой части присваивания")

            assigns: List[PyIRAssign] = []
            prev_value: PyIRNode = rhs_expr

            splits: List[List[tokenize.TokenInfo]] = []
            start = 0
            for idx in eq_indices:
                splits.append(tokens[start:idx])
                start = idx + 1  # пропускаем '='

            # справа налево: c = expr, b = c, a = b
            for lhs_toks in reversed(splits):
                if not lhs_toks:
                    raise SyntaxError("Отсутствует левая часть присваивания")

                stream_l = TokenStream(lhs_toks)
                lhs_expr = StatementBuilder._parse_postfix(stream_l)
                if not stream_l.eof():
                    raise SyntaxError("Слишком сложная цель для присваивания")

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

            return assigns

        # Обычный assignment: LHS = RHS
        eq = eq_indices[0]
        lhs_tokens = tokens[:eq]
        rhs_tokens = tokens[eq + 1 :]

        if not lhs_tokens:
            raise SyntaxError("Отсутствует левая часть присваивания")
        if not rhs_tokens:
            raise SyntaxError("Отсутствует правая часть присваивания")

        lhs_parts = StatementBuilder._split_top_level_commas(lhs_tokens)

        stream_r = TokenStream(rhs_tokens)
        rhs_expr = StatementBuilder._parse_expression(stream_r)
        if not stream_r.eof():
            raise SyntaxError("Некорректное выражение в правой части присваивания")

        # Один таргет: x = expr
        if len(lhs_parts) == 1:
            stream_l = TokenStream(lhs_parts[0])
            lhs_expr = StatementBuilder._parse_postfix(stream_l)
            if not stream_l.eof():
                raise SyntaxError("Некорректная цель присваивания")

            line = lhs_parts[0][0].start[0]
            col = lhs_parts[0][0].start[1]

            return [
                PyIRAssign(
                    line=line,
                    offset=col,
                    targets=[lhs_expr],
                    value=rhs_expr,
                )
            ]

        # Несколько таргетов: a, b = expr
        targets: List[PyIRNode] = []
        for part in lhs_parts:
            stream_l = TokenStream(part)
            t_expr = StatementBuilder._parse_postfix(stream_l)
            if not stream_l.eof():
                raise SyntaxError("Некорректная цель в множественном присваивании")
            targets.append(t_expr)

        line = lhs_parts[0][0].start[0]
        col = lhs_parts[0][0].start[1]

        return [
            PyIRAssign(
                line=line,
                offset=col,
                targets=targets,
                value=rhs_expr,
            )
        ]

    # Expression parser (precedence)
    @staticmethod
    def _parse_expression(stream: TokenStream) -> PyIRNode:
        """
        Верхний уровень выражения:
        - сначала парсим "or"
        - затем, если есть ',' - собираем PyIRTuple (a, b, c)
        """
        first = StatementBuilder._parse_or(stream)

        tok = stream.peek()
        if not (tok and tok.type == tokenize.OP and tok.string == ","):
            return first

        elements: List[PyIRNode] = [first]
        while True:
            tok = stream.peek()
            if not (tok and tok.type == tokenize.OP and tok.string == ","):
                break

            stream.advance()  # съесть ','
            tok2 = stream.peek()
            if tok2 is None:
                raise SyntaxError("Запятая в конце выражения не поддерживается")

            elem = StatementBuilder._parse_or(stream)
            elements.append(elem)

        line = elements[0].line
        col = elements[0].offset
        return PyIRTuple(line=line, offset=col, elements=elements)

    @staticmethod
    def _parse_or(stream: TokenStream) -> PyIRNode:
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
    def _parse_and(stream: TokenStream) -> PyIRNode:
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
    def _parse_compare(stream: TokenStream) -> PyIRNode:
        node = StatementBuilder._parse_bit_or(stream)
        used_comparison = False

        while True:
            tok = stream.peek()
            if tok is None:
                break

            op_enum: PyBinOPType | None = None

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

            elif tok.type == tokenize.NAME:
                if tok.string == "in":
                    stream.advance()
                    op_enum = PyBinOPType.IN

                elif tok.string == "is":
                    stream.advance()
                    nxt = stream.peek()
                    if nxt and nxt.type == tokenize.NAME and nxt.string == "not":
                        stream.advance()
                        op_enum = PyBinOPType.IS_NOT

                    else:
                        op_enum = PyBinOPType.IS

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

            if used_comparison:
                raise SyntaxError(
                    "Цепочные сравнения (a < b < c) не поддерживаются в py2glua"
                )

            used_comparison = True

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
            raise SyntaxError(f"Неизвестный оператор сравнения: {op!r}")

    @staticmethod
    def _parse_bit_or(stream: TokenStream) -> PyIRNode:
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
    def _parse_bit_xor(stream: TokenStream) -> PyIRNode:
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
    def _parse_bit_and(stream: TokenStream) -> PyIRNode:
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
    def _parse_shift(stream: TokenStream) -> PyIRNode:
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
    def _parse_add_sub(stream: TokenStream) -> PyIRNode:
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
    def _parse_mul_div(stream: TokenStream) -> PyIRNode:
        node = StatementBuilder._parse_power(stream)
        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string in {"*", "/", "//", "%"}:
                op_tok = stream.advance()
                assert op_tok is not None
                right = StatementBuilder._parse_power(stream)

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
    def _parse_power(stream: TokenStream) -> PyIRNode:
        """
        '**' — правоассоциативный.
        """
        left = StatementBuilder._parse_unary(stream)

        tok = stream.peek()
        if tok and tok.type == tokenize.OP and tok.string == "**":
            stream.advance()
            right = StatementBuilder._parse_power(stream)  # right-assoc
            return PyIRBinOP(
                line=left.line,
                offset=left.offset,
                op=PyBinOPType.POW,
                left=left,
                right=right,
            )

        return left

    @staticmethod
    def _parse_unary(stream: TokenStream) -> PyIRNode:
        tok = stream.peek()
        if tok is None:
            raise SyntaxError("Неожиданный конец ввода в унарном выражении")

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

    # Primary + postfix (.attr, [idx], (args))
    @staticmethod
    def _parse_postfix(stream: TokenStream) -> PyIRNode:
        node = StatementBuilder._parse_atom(stream)

        while True:
            tok = stream.peek()
            if tok is None:
                break

            if tok.type == tokenize.OP and tok.string == ".":
                stream.advance()
                name_tok = stream.advance()
                if name_tok is None or name_tok.type != tokenize.NAME:
                    raise SyntaxError("Ожидалось имя атрибута после '.'")

                node = PyIRAttribute(
                    line=node.line,
                    offset=node.offset,
                    value=node,
                    attr=name_tok.string,
                )
                continue

            if tok.type == tokenize.OP and tok.string == "(":
                node = StatementBuilder._parse_call(stream, node)
                continue

            if tok.type == tokenize.OP and tok.string == "[":
                node = StatementBuilder._parse_subscript(stream, node)
                continue

            break

        return node

    @staticmethod
    def _parse_atom(stream: TokenStream) -> PyIRNode:
        tok = stream.peek()
        if tok is None:
            raise SyntaxError("Неожиданный конец ввода в выражении")

        # f-string
        if tok.type == tokenize.STRING and tok.string and tok.string[0] in ("f", "F"):
            return StatementBuilder._parse_fstring(stream)

        # Name / keywords treated as names at this stage
        if tok.type == tokenize.NAME:
            tok2 = stream.advance()
            assert tok2 is not None
            line, col = tok2.start
            return PyIRVarUse(line=line, offset=col, name=tok2.string)

        # Constant
        if tok.type in (tokenize.NUMBER, tokenize.STRING):
            tok2 = stream.advance()
            assert tok2 is not None
            line, col = tok2.start
            return PyIRConstant(line=line, offset=col, value=tok2.string)

        # Parenthesized
        if tok.type == tokenize.OP and tok.string == "(":
            lpar = stream.advance()
            assert lpar is not None

            nxt = stream.peek()
            if nxt and nxt.type == tokenize.OP and nxt.string == ")":
                stream.advance()
                line, col = lpar.start
                return PyIRTuple(line=line, offset=col, elements=[])

            node = StatementBuilder._parse_expression(stream)
            stream.expect_op(")")
            return node

        # List
        if tok.type == tokenize.OP and tok.string == "[":
            return StatementBuilder._parse_list_literal(stream)

        # Dict or Set
        if tok.type == tokenize.OP and tok.string == "{":
            return StatementBuilder._parse_dict_or_set_literal(stream)

        raise SyntaxError(f"Неожиданный токен в выражении: {tok!r}")

    @staticmethod
    def _parse_call(stream: TokenStream, func_node: PyIRNode) -> PyIRCall:
        lpar = stream.advance()
        assert lpar and lpar.string == "("

        args_p: list[PyIRNode] = []
        args_kw: dict[str, PyIRNode] = {}

        tok = stream.peek()
        if tok and tok.type == tokenize.OP and tok.string in ("*", "**"):
            raise SyntaxError(
                "Распаковка аргументов (*args / **kwargs) не поддерживается в py2glua"
            )

        if not (tok and tok.type == tokenize.OP and tok.string == ")"):
            while True:
                t = stream.peek()
                if t and t.type == tokenize.OP and t.string in ("*", "**"):
                    raise SyntaxError(
                        "Распаковка аргументов (*args / **kwargs) не поддерживается в py2glua"
                    )

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
                    stream.advance()  # '='
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

        return PyIRCall(
            line=func_node.line,
            offset=func_node.offset,
            func=func_node,
            args_p=args_p,
            args_kw=args_kw,
        )

    @staticmethod
    def _parse_subscript(stream: TokenStream, base: PyIRNode) -> PyIRSubscript:
        lbr = stream.advance()
        assert lbr and lbr.string == "["

        index_expr = StatementBuilder._parse_expression(stream)
        stream.expect_op("]")

        return PyIRSubscript(
            line=base.line,
            offset=base.offset,
            value=base,
            index=index_expr,
        )

    @staticmethod
    def _parse_list_literal(stream: TokenStream) -> PyIRList:
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
                    tok2 = stream.peek()
                    if tok2 and tok2.type == tokenize.OP and tok2.string == "]":
                        break
                    continue
                break

        stream.expect_op("]")
        line, col = lbr.start
        return PyIRList(line=line, offset=col, elements=elements)

    @staticmethod
    def _parse_dict_or_set_literal(stream: TokenStream) -> PyIRNode:
        lbr = stream.advance()
        assert lbr and lbr.string == "{"

        line, col = lbr.start

        tok = stream.peek()
        if tok and tok.type == tokenize.OP and tok.string == "}":
            stream.advance()
            return PyIRDict(line=line, offset=col, items=[])

        key = StatementBuilder._parse_expression_until(stream, {":"})
        tok = stream.peek()

        if tok and tok.type == tokenize.OP and tok.string == ":":
            items: list[PyIRDictItem] = []

            while True:
                stream.expect_op(":")
                value = StatementBuilder._parse_expression_until(stream, {",", "}"})

                items.append(
                    PyIRDictItem(
                        line=key.line,
                        offset=key.offset,
                        key=key,
                        value=value,
                    )
                )

                tok = stream.peek()
                if tok and tok.type == tokenize.OP and tok.string == ",":
                    stream.advance()

                    tok = stream.peek()
                    if tok and tok.type == tokenize.OP and tok.string == "}":
                        stream.advance()
                        return PyIRDict(line=line, offset=col, items=items)

                    key = StatementBuilder._parse_expression_until(stream, {":"})
                    continue

                if tok and tok.type == tokenize.OP and tok.string == "}":
                    stream.advance()
                    return PyIRDict(line=line, offset=col, items=items)

                raise SyntaxError(f"Expected ',' or '}}' in dict literal, got {tok!r}")

        elements: list[PyIRNode] = [key]

        while True:
            tok = stream.peek()
            if tok and tok.type == tokenize.OP and tok.string == ",":
                stream.advance()

                tok = stream.peek()
                if tok and tok.type == tokenize.OP and tok.string == "}":
                    stream.advance()
                    return PyIRSet(line=line, offset=col, elements=elements)

                el = StatementBuilder._parse_expression(stream)
                elements.append(el)
                continue

            if tok and tok.type == tokenize.OP and tok.string == "}":
                stream.advance()
                return PyIRSet(line=line, offset=col, elements=elements)

            raise SyntaxError(f"Expected ',' or '}}' in set literal, got {tok!r}")

    @staticmethod
    def _parse_fstring(stream: TokenStream) -> PyIRFString:
        tok = stream.advance()
        assert tok is not None

        raw = tok.string
        if not raw or raw[0] not in ("f", "F"):
            raise SyntaxError("Некорректная f-строка")

        # минималистичная поддержка: f"....{name}...."
        # без форматов, без выражений, только идентификатор
        if not raw.startswith(('f"', "f'", 'F"', "F'")):
            raise SyntaxError(
                "Поддерживаются только простые f-строки вида f\"...\" или f'...'"
            )

        body = raw[2:-1]

        parts: list[str | PyIRNode] = []
        buf: list[str] = []

        i = 0
        n = len(body)

        def flush() -> None:
            if buf:
                parts.append("".join(buf))
                buf.clear()

        while i < n:
            c = body[i]

            if c == "{":
                flush()
                i += 1
                start = i

                while i < n and body[i] != "}":
                    i += 1

                if i >= n:
                    raise SyntaxError("Незакрытая '{' в f-строке")

                name = body[start:i].strip()
                if not name.isidentifier():
                    raise SyntaxError(
                        "В f-строках разрешена только подстановка простого имени: {name}"
                    )

                parts.append(
                    PyIRVarUse(
                        line=tok.start[0],
                        offset=tok.start[1],
                        name=name,
                    )
                )
                i += 1
                continue

            buf.append(c)
            i += 1

        flush()

        return PyIRFString(
            line=tok.start[0],
            offset=tok.start[1],
            parts=parts,
        )

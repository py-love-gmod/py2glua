import tokenize
from pathlib import Path

from ..py_logic_block_builder import (
    PyLogicBlockBuilder,
    PyLogicKind,
    PyLogicNode,
)
from .import_analyzer import ImportAnalyzer
from .py_ir_dataclass import (
    PyIRCall,
    PyIRClassDef,
    PyIRComment,
    PyIRContext,
    PyIRDecorator,
    PyIRDel,
    PyIRExceptHandler,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
    PyIRTry,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from .statement_compiler import StatementCompiler


class PyIRBuilder:
    @classmethod
    def build_file(cls, source: str, path_to_file: Path | None = None) -> PyIRFile:
        logic_blocks = PyLogicBlockBuilder.build(source)

        context = PyIRContext(
            parent_context=None,
            meta={
                "aliases": {},
                "imports": [],
            },
        )

        py_ir_file = PyIRFile(
            line=None,
            offset=None,
            path=path_to_file,
            context=context,
        )
        py_ir_file.body = cls._build_ir_block(logic_blocks, py_ir_file)
        return py_ir_file

    # region Core block dispatcher
    @classmethod
    def _build_ir_block(
        cls,
        nodes: list[PyLogicNode],
        parent_obj: PyIRNode,
    ) -> list[PyIRNode]:
        dispatch = {
            PyLogicKind.FUNCTION: cls._build_ir_function,
            PyLogicKind.CLASS: cls._build_ir_class,
            PyLogicKind.BRANCH: cls._build_ir_branch,
            PyLogicKind.LOOP: cls._build_ir_loop,
            PyLogicKind.TRY: cls._build_ir_try,
            PyLogicKind.WITH: cls._build_ir_with,
            PyLogicKind.IMPORT: ImportAnalyzer.build,
            PyLogicKind.DELETE: cls._build_ir_delete,
            PyLogicKind.RETURN: cls._build_ir_return,
            PyLogicKind.PASS: cls._build_ir_pass,
            PyLogicKind.COMMENT: cls._build_ir_comment,
            PyLogicKind.STATEMENT: cls._build_ir_statement,
        }

        out: list[PyIRNode] = []
        for node in nodes:
            func = dispatch.get(node.kind)
            if func is None:
                raise ValueError(
                    f"PyLogicKind {node.kind} has no handler in PyIRBuilder"
                )

            result_nodes = func(parent_obj, node)
            out.extend(result_nodes)

        return out

    # endregion

    # region Helpers

    @staticmethod
    def _tokens_from_origin(origin) -> list[tokenize.TokenInfo]:
        return [t for t in origin.tokens if isinstance(t, tokenize.TokenInfo)]

    @classmethod
    def _first_token_from_origins(cls, origins) -> tokenize.TokenInfo | None:
        for origin in origins:
            toks = cls._tokens_from_origin(origin)
            if toks:
                return toks[0]

        return None

    @classmethod
    def _first_line_from_origins(cls, origins) -> int:
        tok = cls._first_token_from_origins(origins)
        return tok.start[0] if tok is not None else 0

    @staticmethod
    def _extract_name_from_header(
        header_origin,
        keyword: str,
    ) -> tuple[str, int, int]:
        tokens = [t for t in header_origin.tokens if isinstance(t, tokenize.TokenInfo)]
        if not tokens:
            raise SyntaxError("Empty header in logic node")

        first = tokens[0]
        name: str | None = None
        seen_kw = False

        for tok in tokens:
            if tok.string == keyword:
                seen_kw = True
                continue

            if seen_kw and tok.type == tokenize.NAME:
                name = tok.string
                break

        if name is None:
            raise SyntaxError(f"Cannot find name after {keyword} in header")

        return name, first.start[0], first.start[1]

    @staticmethod
    def _extract_header_expr_tokens(
        header_origin,
        keyword: str,
    ) -> list[tokenize.TokenInfo]:
        tokens = [t for t in header_origin.tokens if isinstance(t, tokenize.TokenInfo)]
        if not tokens:
            return []

        kw_idx = None
        colon_idx = None

        for i, tok in enumerate(tokens):
            if kw_idx is None and tok.string == keyword:
                kw_idx = i
            if tok.string == ":":
                colon_idx = i
                break

        if kw_idx is None:
            raise SyntaxError(f"Header does not contain '{keyword}' keyword")

        if colon_idx is None:
            colon_idx = len(tokens)

        return tokens[kw_idx + 1 : colon_idx]

    @staticmethod
    def _split_for_header_tokens(
        header_origin,
    ) -> tuple[list[tokenize.TokenInfo], list[tokenize.TokenInfo]]:
        tokens = [t for t in header_origin.tokens if isinstance(t, tokenize.TokenInfo)]
        if not tokens:
            raise SyntaxError("Empty for-header")

        for_idx = None
        in_idx = None
        colon_idx = None

        for i, tok in enumerate(tokens):
            if for_idx is None and tok.string == "for":
                for_idx = i

            elif tok.string == "in" and for_idx is not None and in_idx is None:
                in_idx = i

            elif tok.string == ":":
                colon_idx = i
                break

        if for_idx is None or in_idx is None:
            raise SyntaxError("Malformed 'for' header")

        if colon_idx is None:
            colon_idx = len(tokens)

        target_tokens = tokens[for_idx + 1 : in_idx]
        iter_tokens = tokens[in_idx + 1 : colon_idx]

        if not target_tokens or not iter_tokens:
            raise SyntaxError("Malformed 'for' header: empty target or iterable")

        return target_tokens, iter_tokens

    @staticmethod
    def _build_ir_decorators(
        parent_obj: PyIRNode,
        decorator_origins: list,
    ) -> list[PyIRDecorator]:
        decorators: list[PyIRDecorator] = []

        for origin in decorator_origins:
            tokens = [
                t
                for t in origin.tokens
                if isinstance(t, tokenize.TokenInfo)
                and t.type not in (tokenize.NL, tokenize.NEWLINE)
            ]
            if not tokens:
                continue

            first = tokens[0]

            if tokens and tokens[0].string == "@":
                tokens = tokens[1:]

            if not tokens:
                continue

            name_tokens: list[tokenize.TokenInfo] = []
            has_call = False
            for tok in tokens:
                if tok.string == "(":
                    has_call = True
                    break

                name_tokens.append(tok)

            name_str = "".join(tok.string for tok in name_tokens).strip()

            args_p: list[PyIRNode] = []
            args_kw: dict[str, PyIRNode] = {}

            if has_call:
                try:
                    ir_expr = StatementCompiler.compile_expres(tokens, parent_obj)

                except Exception:
                    ir_expr = None

                else:
                    if isinstance(ir_expr, PyIRCall):
                        if ir_expr.name:
                            name_str = ir_expr.name

                        args_p = ir_expr.args_p
                        args_kw = ir_expr.args_kw

            decorators.append(
                PyIRDecorator(
                    line=first.start[0],
                    offset=first.start[1],
                    name=name_str,
                    args_p=args_p,
                    args_kw=args_kw,
                )
            )

        return decorators

    @classmethod
    def _partition_children_by_headers(
        cls,
        headers: list,
        children: list[PyLogicNode],
    ) -> list[list[PyLogicNode]]:
        if not headers:
            return [[]]

        header_lines = [cls._first_line_from_origins([h]) for h in headers]
        clause_children: list[list[PyLogicNode]] = [[] for _ in headers]

        for child in children:
            child_line = cls._first_line_from_origins(child.origins)
            idx = 0
            for i in range(len(headers) - 1, -1, -1):
                if child_line >= header_lines[i]:
                    idx = i
                    break
            clause_children[idx].append(child)

        return clause_children

    @classmethod
    def _parse_except_header(
        cls,
        header_origin,
        parent_obj: PyIRNode,
    ) -> tuple[PyIRNode | None, str | None]:
        tokens = [t for t in header_origin.tokens if isinstance(t, tokenize.TokenInfo)]
        if not tokens:
            raise SyntaxError("Empty except header")

        kw_idx = None
        colon_idx = None
        for i, tok in enumerate(tokens):
            if kw_idx is None and tok.string == "except":
                kw_idx = i

            if tok.string == ":":
                colon_idx = i
                break

        if kw_idx is None:
            raise SyntaxError("Malformed except header: no 'except' keyword")

        if colon_idx is None:
            colon_idx = len(tokens)

        inner = tokens[kw_idx + 1 : colon_idx]
        if not inner:
            return None, None

        as_idx = None
        for i, tok in enumerate(inner):
            if tok.string == "as":
                as_idx = i
                break

        type_tokens: list[tokenize.TokenInfo] = []
        name: str | None = None

        if as_idx is None:
            type_tokens = inner
        else:
            type_tokens = inner[:as_idx]
            name_tokens = inner[as_idx + 1 :]
            for t in name_tokens:
                if t.type == tokenize.NAME:
                    name = t.string
                    break

        exc_type_ir: PyIRNode | None = None
        if type_tokens:
            exc_type_ir = StatementCompiler.compile_expres(type_tokens, parent_obj)

        return exc_type_ir, name

    @classmethod
    def _parse_with_items(
        cls,
        header_origin,
        parent_obj: PyIRNode,
    ) -> list[PyIRWithItem]:
        tokens = [t for t in header_origin.tokens if isinstance(t, tokenize.TokenInfo)]
        if not tokens:
            raise SyntaxError("Empty with header")

        with_idx = None
        colon_idx = None
        for i, tok in enumerate(tokens):
            if with_idx is None and tok.string == "with":
                with_idx = i
            if tok.string == ":":
                colon_idx = i
                break

        if with_idx is None:
            raise SyntaxError("Malformed with header: no 'with' keyword")

        if colon_idx is None:
            colon_idx = len(tokens)

        inner = tokens[with_idx + 1 : colon_idx]

        if any(t.string == "," for t in inner):
            raise SyntaxError("Multiple with-items are not supported in py2glua yet")

        as_idx = None
        for i, tok in enumerate(inner):
            if tok.string == "as":
                as_idx = i
                break

        ctx_tokens: list[tokenize.TokenInfo]
        var_tokens: list[tokenize.TokenInfo] | None = None

        if as_idx is None:
            ctx_tokens = inner

        else:
            ctx_tokens = inner[:as_idx]
            var_tokens = inner[as_idx + 1 :] or None

        first = tokens[0]

        ctx_ir = StatementCompiler.compile_expres(ctx_tokens, parent_obj)
        var_ir: PyIRNode | None = None
        if var_tokens:
            var_ir = StatementCompiler.compile_expres(var_tokens, parent_obj)

        item = PyIRWithItem(
            line=first.start[0],
            offset=first.start[1],
            context_expr=ctx_ir,
            optional_vars=var_ir,
        )
        return [item]

    # endregion

    # region FUNCTION / CLASS
    @classmethod
    def _build_ir_function(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        if not node.origins:
            raise ValueError("Function logic node has no origins")

        header_origin = node.origins[-1]
        decorator_origins = node.origins[:-1]

        func_name, line, offset = cls._extract_name_from_header(
            header_origin,
            keyword="def",
        )

        parent_ctx = getattr(parent_obj, "context", None)
        func_ctx = PyIRContext(
            parent_context=parent_ctx,
            meta={
                "aliases": {},
                "imports": [],
            },
        )

        decorators = cls._build_ir_decorators(parent_obj, decorator_origins)

        ir_func = PyIRFunctionDef(
            line=line,
            offset=offset,
            name=func_name,
            signature={},  # TODO: нормальный парсинг сигнатуры
            context=func_ctx,
            decorators=decorators,
            body=[],
        )

        ir_body = cls._build_ir_block(node.children, ir_func)
        ir_func.body = ir_body

        return [ir_func]

    @classmethod
    def _build_ir_class(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        if not node.origins:
            raise ValueError("Class logic node has no origins")

        header_origin = node.origins[-1]
        decorator_origins = node.origins[:-1]

        class_name, line, offset = cls._extract_name_from_header(
            header_origin,
            keyword="class",
        )

        parent_ctx = getattr(parent_obj, "context", None)
        class_ctx = PyIRContext(
            parent_context=parent_ctx,
            meta={
                "aliases": {},
                "imports": [],
            },
        )

        decorators = cls._build_ir_decorators(parent_obj, decorator_origins)

        ir_class = PyIRClassDef(
            line=line,
            offset=offset,
            name=class_name,
            context=class_ctx,
            decorators=decorators,
            body=[],
        )

        ir_body = cls._build_ir_block(node.children, ir_class)
        ir_class.body = ir_body

        return [ir_class]

    # endregion

    # region BRANCH / LOOP / TRY / WITH
    @classmethod
    def _build_ir_branch(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        if not node.origins:
            raise ValueError("Branch logic node has no origins")

        headers = node.origins
        clause_children = cls._partition_children_by_headers(headers, node.children)

        first_header = headers[0]
        first_tokens = cls._tokens_from_origin(first_header)
        if not first_tokens:
            raise SyntaxError("Empty if header")

        first_tok = first_tokens[0]
        test_tokens = cls._extract_header_expr_tokens(first_header, "if")
        test_ir = StatementCompiler.compile_expres(test_tokens, parent_obj)

        ir_if = PyIRIf(
            line=first_tok.start[0],
            offset=first_tok.start[1],
            test=test_ir,
            body=cls._build_ir_block(clause_children[0], parent_obj),
            orelse=[],
        )

        current_if = ir_if

        for hdr, body_nodes in zip(headers[1:], clause_children[1:]):
            toks = cls._tokens_from_origin(hdr)
            if not toks:
                continue

            kw = toks[0].string
            if kw == "elif":
                cond_tokens = cls._extract_header_expr_tokens(hdr, "elif")
                cond_ir = StatementCompiler.compile_expres(cond_tokens, parent_obj)

                new_if = PyIRIf(
                    line=toks[0].start[0],
                    offset=toks[0].start[1],
                    test=cond_ir,
                    body=cls._build_ir_block(body_nodes, parent_obj),
                    orelse=[],
                )
                current_if.orelse = [new_if]
                current_if = new_if

            elif kw == "else":
                current_if.orelse = cls._build_ir_block(body_nodes, parent_obj)

            else:
                raise SyntaxError(f"Unexpected branch keyword {kw!r} in BRANCH node")

        return [ir_if]

    @classmethod
    def _build_ir_loop(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        if not node.origins:
            raise ValueError("Loop logic node has no origins")

        header_origin = node.origins[0]
        tokens = cls._tokens_from_origin(header_origin)
        if not tokens:
            raise SyntaxError("Empty loop header")

        first_tok = tokens[0]
        kw = first_tok.string

        if kw == "while":
            cond_tokens = cls._extract_header_expr_tokens(header_origin, "while")
            cond_ir = StatementCompiler.compile_expres(cond_tokens, parent_obj)
            body_ir = cls._build_ir_block(node.children, parent_obj)
            ir_while = PyIRWhile(
                line=first_tok.start[0],
                offset=first_tok.start[1],
                test=cond_ir,
                body=body_ir,
                orelse=[],
            )
            return [ir_while]

        if kw == "for":
            target_tokens, iter_tokens = cls._split_for_header_tokens(header_origin)
            target_ir = StatementCompiler.compile_expres(target_tokens, parent_obj)
            iter_ir = StatementCompiler.compile_expres(iter_tokens, parent_obj)
            body_ir = cls._build_ir_block(node.children, parent_obj)
            ir_for = PyIRFor(
                line=first_tok.start[0],
                offset=first_tok.start[1],
                target=target_ir,
                iter=iter_ir,
                body=body_ir,
                orelse=[],
            )
            return [ir_for]

        raise SyntaxError(f"Unexpected loop keyword {kw!r}")

    @classmethod
    def _build_ir_try(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        if not node.origins:
            raise ValueError("Try logic node has no origins")

        headers = node.origins
        clause_children = cls._partition_children_by_headers(headers, node.children)

        try_header = headers[0]
        toks = cls._tokens_from_origin(try_header)
        if not toks:
            raise SyntaxError("Empty try header")

        first_tok = toks[0]

        ir_try = PyIRTry(
            line=first_tok.start[0],
            offset=first_tok.start[1],
            body=[],
            handlers=[],
            orelse=[],
            finalbody=[],
        )

        ir_try.body = cls._build_ir_block(clause_children[0], parent_obj)

        for hdr, body_nodes in zip(headers[1:], clause_children[1:]):
            htoks = cls._tokens_from_origin(hdr)
            if not htoks:
                continue

            kw = htoks[0].string

            if kw == "except":
                exc_type_ir, name = cls._parse_except_header(hdr, parent_obj)
                handler = PyIRExceptHandler(
                    line=htoks[0].start[0],
                    offset=htoks[0].start[1],
                    type=exc_type_ir,
                    name=name,
                    body=cls._build_ir_block(body_nodes, parent_obj),
                )
                ir_try.handlers.append(handler)

            elif kw == "else":
                ir_try.orelse = cls._build_ir_block(body_nodes, parent_obj)

            elif kw == "finally":
                ir_try.finalbody = cls._build_ir_block(body_nodes, parent_obj)

            else:
                raise SyntaxError(f"Unexpected try-chain keyword {kw!r}")

        return [ir_try]

    @classmethod
    def _build_ir_with(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        if not node.origins:
            raise ValueError("With logic node has no origins")

        header_origin = node.origins[0]
        tokens = cls._tokens_from_origin(header_origin)
        if not tokens:
            raise SyntaxError("Empty with header")

        first_tok = tokens[0]

        items = cls._parse_with_items(header_origin, parent_obj)
        body_ir = cls._build_ir_block(node.children, parent_obj)

        ir_with = PyIRWith(
            line=first_tok.start[0],
            offset=first_tok.start[1],
            items=items,
            body=body_ir,
        )
        return [ir_with]

    # endregion

    # region Simple statements
    @classmethod
    def _build_ir_statement(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origins[0]
        ir_node = StatementCompiler.compile_expres(origin.tokens, parent_obj)
        return [ir_node]

    @classmethod
    def _build_ir_return(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origins[0]
        header_tokens = [t for t in origin.tokens if isinstance(t, tokenize.TokenInfo)]
        if not header_tokens:
            raise SyntaxError("Empty return statement")

        first_token = header_tokens[0]
        tail_tokens = origin.tokens[1:]
        ir = StatementCompiler.compile_expres(tail_tokens, parent_obj)
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
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origins[0]
        header_tokens = [t for t in origin.tokens if isinstance(t, tokenize.TokenInfo)]
        if not header_tokens:
            raise SyntaxError("Empty del statement")

        first_token = header_tokens[0]
        tail_tokens = origin.tokens[1:]
        ir = StatementCompiler.compile_expres(tail_tokens, parent_obj)
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
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        origin = node.origins[0]
        header_tokens = [t for t in origin.tokens if isinstance(t, tokenize.TokenInfo)]
        if not header_tokens:
            raise SyntaxError("Empty pass statement")

        first_token = header_tokens[0]
        return [
            PyIRPass(
                first_token.start[0],
                first_token.start[1],
            )
        ]

    @classmethod
    def _build_ir_comment(
        cls,
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> list[PyIRNode]:
        text = ""
        first_token: tokenize.TokenInfo | None = None

        for origin in node.origins:
            for tok in origin.tokens:
                if not isinstance(tok, tokenize.TokenInfo):
                    continue

                if first_token is None:
                    first_token = tok

                text += tok.string.removeprefix("# ")

        if first_token is None:
            first_line, first_offset = 0, 0
        else:
            first_line, first_offset = first_token.start

        return [
            PyIRComment(
                first_line,
                first_offset,
                text,
            )
        ]

    # endregion

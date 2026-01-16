import tokenize
from typing import List, Sequence

from ...etc import TokenStream
from ...parse import PyLogicKind, PyLogicNode
from ..build_context import build_block
from ..ir_dataclass import PyIRFunctionDef, PyIRNode
from .statement_builder import StatementBuilder


class FunctionBuilder:
    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if node.kind is not PyLogicKind.FUNCTION:
            raise ValueError("FunctionBuilder expects PyLogicKind.FUNCTION")

        if not node.origins:
            raise ValueError("PyLogicNode.FUNCTION has no origins")

        header = node.origins[0]

        tokens: List[tokenize.TokenInfo] = [
            t
            for t in header.tokens
            if isinstance(t, tokenize.TokenInfo)
            and t.type
            not in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            )
        ]

        if not tokens:
            raise SyntaxError("Empty function header")

        def_idx = FunctionBuilder._find_def(tokens)

        if def_idx + 1 >= len(tokens) or tokens[def_idx + 1].type != tokenize.NAME:
            raise SyntaxError("Expected function name after 'def'")

        name_tok = tokens[def_idx + 1]
        func_name = name_tok.string

        if def_idx + 2 >= len(tokens) or not (
            tokens[def_idx + 2].type == tokenize.OP
            and tokens[def_idx + 2].string == "("
        ):
            raise SyntaxError("Expected '(' after function name")

        lpar_idx = def_idx + 2
        rpar_idx = FunctionBuilder._find_matching_paren(tokens, lpar_idx)

        params_tokens = tokens[lpar_idx + 1 : rpar_idx]

        tail = tokens[rpar_idx + 1 :]
        if not tail:
            raise SyntaxError("Function header must end with ':'")

        return_ann_str: str | None = None
        colon_idx_in_tail = FunctionBuilder._find_colon_in_tail(tail)

        before_colon = tail[:colon_idx_in_tail]
        after_colon = tail[colon_idx_in_tail + 1 :]

        if after_colon:
            raise SyntaxError("Unexpected tokens after ':' in function header")

        if before_colon:
            if not (
                len(before_colon) >= 2 and FunctionBuilder._is_op(before_colon[0], "->")
            ):
                raise SyntaxError("Only return annotation '-> T' is allowed after ')'")

            ann_tokens = before_colon[1:]
            if not ann_tokens:
                raise SyntaxError("Missing return annotation after '->'")

            return_ann_str = FunctionBuilder._tokens_to_clean_src(ann_tokens).strip()
            if not return_ann_str:
                raise SyntaxError("Empty return annotation")

        signature = FunctionBuilder._parse_params(params_tokens)
        if return_ann_str is not None:
            signature["__return__"] = (return_ann_str, None)

        body_children = list(node.children)

        body = build_block(body_children)

        line, col = tokens[def_idx].start

        return [
            PyIRFunctionDef(
                line=line,
                offset=col,
                name=func_name,
                signature=signature,
                decorators=[],
                body=body,
            )
        ]

    # region helpers
    @staticmethod
    def _find_def(tokens: List[tokenize.TokenInfo]) -> int:
        for i, t in enumerate(tokens):
            if t.type == tokenize.NAME and t.string == "def":
                return i

        raise SyntaxError("Expected 'def' in function header")

    @staticmethod
    def _find_matching_paren(tokens: List[tokenize.TokenInfo], lpar_idx: int) -> int:
        if not (
            tokens[lpar_idx].type == tokenize.OP and tokens[lpar_idx].string == "("
        ):
            raise ValueError("lpar_idx must point to '('")

        depth = 0
        for i in range(lpar_idx, len(tokens)):
            t = tokens[i]
            if t.type == tokenize.OP and t.string == "(":
                depth += 1

            elif t.type == tokenize.OP and t.string == ")":
                depth -= 1
                if depth == 0:
                    return i

        raise SyntaxError("Unmatched '(' in function signature")

    @staticmethod
    def _find_colon_in_tail(tail: List[tokenize.TokenInfo]) -> int:
        for i, t in enumerate(tail):
            if t.type == tokenize.OP and t.string == ":":
                return i

        raise SyntaxError("Function header must end with ':'")

    @staticmethod
    def _is_op(tok: tokenize.TokenInfo, s: str) -> bool:
        return tok.type == tokenize.OP and tok.string == s

    @staticmethod
    def _tokens_to_clean_src(tokens: list[tokenize.TokenInfo]) -> str:
        parts: list[str] = []

        for t in tokens:
            if t.type in (tokenize.NAME, tokenize.OP, tokenize.STRING, tokenize.NUMBER):
                parts.append(t.string)

        return "".join(parts)

    @staticmethod
    def _split_top_level_commas(
        tokens: List[tokenize.TokenInfo],
    ) -> List[List[tokenize.TokenInfo]]:
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

        if len(out) == 1 and not out[0]:
            return []

        return out

    @staticmethod
    def _parse_params(
        tokens: List[tokenize.TokenInfo],
    ) -> dict[str, tuple[str | None, str | None]]:
        signature: dict[str, tuple[str | None, str | None]] = {}

        if not tokens:
            return signature

        for t in tokens:
            if t.type == tokenize.OP and t.string in {"/", "*"}:  # TODO:
                raise SyntaxError(
                    "Keyword-only/posonly markers (*, /) are not supported in py2glua"
                )

            if t.type == tokenize.OP and t.string == "**":  # TODO:
                raise SyntaxError("**kwargs is not supported in py2glua")

            if t.type == tokenize.OP and t.string == "*":  # TODO:
                raise SyntaxError("*args is not supported in py2glua")

        parts = FunctionBuilder._split_top_level_commas(tokens)
        for part in parts:
            part = [t for t in part if not (t.type == tokenize.OP and t.string == ",")]
            if not part:
                raise SyntaxError("Empty parameter in function signature")

            name, ann_tokens, default_tokens = FunctionBuilder._parse_single_param(part)

            if name in signature:
                raise SyntaxError(f"Duplicate parameter name: {name!r}")

            ann_str = (
                FunctionBuilder._tokens_to_clean_src(ann_tokens).strip()
                if ann_tokens
                else None
            )

            default_str: str | None = None
            if default_tokens is not None:
                stream = TokenStream(default_tokens)
                _ = StatementBuilder._parse_expression(stream)
                if not stream.eof():
                    raise SyntaxError(
                        "Invalid default value expression in function parameter"
                    )

                default_str = FunctionBuilder._tokens_to_clean_src(
                    default_tokens
                ).strip()
                if not default_str:
                    raise SyntaxError("Empty default expression in function parameter")

            signature[name] = (ann_str, default_str)

        return signature

    @staticmethod
    def _parse_single_param(
        tokens: List[tokenize.TokenInfo],
    ) -> tuple[str, List[tokenize.TokenInfo], List[tokenize.TokenInfo] | None]:
        if tokens[0].type == tokenize.OP and tokens[0].string in {"*", "**"}:
            raise SyntaxError(
                "Argument unpacking in function signature is not supported in py2glua"
            )

        if tokens[0].type != tokenize.NAME:
            raise SyntaxError("Expected parameter name")

        name = tokens[0].string
        rest = tokens[1:]

        if not rest:
            return name, [], None

        colon_idx: int | None = None
        eq_idx: int | None = None

        depth = 0
        for i, t in enumerate(rest):
            if t.type == tokenize.OP:
                if t.string in "([{":
                    depth += 1
                elif t.string in ")]}":
                    depth -= 1

            if depth != 0:
                continue

            if colon_idx is None and t.type == tokenize.OP and t.string == ":":
                colon_idx = i
                continue

            if eq_idx is None and t.type == tokenize.OP and t.string == "=":
                eq_idx = i
                continue

        ann_tokens: List[tokenize.TokenInfo] = []
        default_tokens: List[tokenize.TokenInfo] | None = None

        if colon_idx is None and eq_idx is not None:
            rhs = rest[eq_idx + 1 :]
            if not rhs:
                raise SyntaxError("Missing default value in parameter")
            default_tokens = rhs
            return name, ann_tokens, default_tokens

        if colon_idx is not None and eq_idx is None:
            ann = rest[colon_idx + 1 :]
            if not ann:
                raise SyntaxError("Missing annotation after ':' in parameter")

            ann_tokens = ann
            return name, ann_tokens, None

        if colon_idx is not None and eq_idx is not None:
            if eq_idx < colon_idx:
                raise SyntaxError("Invalid parameter syntax: '=' before ':'")

            ann = rest[colon_idx + 1 : eq_idx]
            if not ann:
                raise SyntaxError("Missing annotation after ':' in parameter")

            rhs = rest[eq_idx + 1 :]
            if not rhs:
                raise SyntaxError("Missing default value in parameter")

            ann_tokens = ann
            default_tokens = rhs
            return name, ann_tokens, default_tokens

        raise SyntaxError("Invalid parameter syntax")

    # endregion

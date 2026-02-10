from __future__ import annotations

import token as token_mod
import tokenize
from typing import List, NoReturn, Sequence

from ...etc import TokenStream
from ...parse import PyLogicKind, PyLogicNode
from ..build_context import build_block
from ..ir_dataclass import PyIRFunctionDef, PyIRNode
from .statement_builder import StatementBuilder


class FunctionBuilder:
    @staticmethod
    def _raise(msg: str, tok: tokenize.TokenInfo | None) -> NoReturn:
        if tok is None:
            raise SyntaxError(msg)

        line, col = tok.start
        raise SyntaxError(f"{msg}\nLINE|OFFSET: {line}|{col}")

    @staticmethod
    def build(node: PyLogicNode) -> Sequence[PyIRNode]:
        if node.kind is not PyLogicKind.FUNCTION:
            raise ValueError("FunctionBuilder ожидает PyLogicKind.FUNCTION")

        if not node.origins:
            raise ValueError("У PyLogicNode.FUNCTION отсутствуют origins")

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
            raise SyntaxError("Пустой заголовок функции")

        def_idx = FunctionBuilder._find_def(tokens)

        if def_idx + 1 >= len(tokens) or tokens[def_idx + 1].type != tokenize.NAME:
            FunctionBuilder._raise(
                "Ожидалось имя функции после 'def'",
                tokens[def_idx] if def_idx < len(tokens) else None,
            )

        name_tok = tokens[def_idx + 1]
        func_name = name_tok.string

        if def_idx + 2 >= len(tokens) or not (
            tokens[def_idx + 2].type == tokenize.OP
            and tokens[def_idx + 2].string == "("
        ):
            FunctionBuilder._raise("Ожидалась '(' после имени функции", name_tok)

        lpar_idx = def_idx + 2
        rpar_idx = FunctionBuilder._find_matching_paren(tokens, lpar_idx)

        params_tokens = tokens[lpar_idx + 1 : rpar_idx]

        tail = tokens[rpar_idx + 1 :]
        if not tail:
            FunctionBuilder._raise(
                "Заголовок функции должен заканчиваться ':'",
                tokens[rpar_idx],
            )

        return_ann_str: str | None = None
        colon_idx_in_tail = FunctionBuilder._find_colon_in_tail(tail)

        before_colon = tail[:colon_idx_in_tail]
        after_colon = tail[colon_idx_in_tail + 1 :]

        if after_colon:
            FunctionBuilder._raise(
                "Лишние токены после ':' в заголовке функции",
                after_colon[0],
            )

        if before_colon:
            if not (
                len(before_colon) >= 2 and FunctionBuilder._is_op(before_colon[0], "->")
            ):
                FunctionBuilder._raise(
                    "После ')' допускается только аннотация возвращаемого типа '-> T'",
                    before_colon[0],
                )

            ann_tokens = before_colon[1:]
            if not ann_tokens:
                FunctionBuilder._raise(
                    "Отсутствует аннотация возвращаемого типа после '->'",
                    before_colon[0],
                )

            return_ann_str = FunctionBuilder._tokens_to_clean_src(ann_tokens).strip()
            if not return_ann_str:
                FunctionBuilder._raise("Пустая аннотация возвращаемого типа", ann_tokens[0])

        signature, vararg_name, kwarg_name = FunctionBuilder._parse_params(params_tokens)

        body_children = list(node.children)
        body = build_block(body_children)

        line, col = tokens[def_idx].start

        return [
            PyIRFunctionDef(
                line=line,
                offset=col,
                name=func_name,
                signature=signature,
                returns=return_ann_str,
                vararg=vararg_name,
                kwarg=kwarg_name,
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

        raise SyntaxError("Ожидался 'def' в заголовке функции")

    @staticmethod
    def _find_matching_paren(tokens: List[tokenize.TokenInfo], lpar_idx: int) -> int:
        if not (
            tokens[lpar_idx].type == tokenize.OP and tokens[lpar_idx].string == "("
        ):
            raise ValueError("lpar_idx должен указывать на '('")

        depth = 0
        for i in range(lpar_idx, len(tokens)):
            t = tokens[i]
            if t.type == tokenize.OP and t.string == "(":
                depth += 1

            elif t.type == tokenize.OP and t.string == ")":
                depth -= 1
                if depth == 0:
                    return i

        raise SyntaxError("Не найдена закрывающая ')' в сигнатуре функции")

    @staticmethod
    def _find_colon_in_tail(tail: List[tokenize.TokenInfo]) -> int:
        for i, t in enumerate(tail):
            if t.type == tokenize.OP and t.string == ":":
                return i

        raise SyntaxError("Заголовок функции должен заканчиваться ':'")

    @staticmethod
    def _is_op(tok: tokenize.TokenInfo, s: str) -> bool:
        return tok.type == tokenize.OP and tok.string == s

    @staticmethod
    def _tokens_to_clean_src(tokens: list[tokenize.TokenInfo]) -> str:
        ellipsis_type = getattr(token_mod, "ELLIPSIS", None)
        parts: list[str] = []
        for t in tokens:
            if t.type in (tokenize.NAME, tokenize.OP, tokenize.STRING, tokenize.NUMBER):
                parts.append(t.string)
                continue
            if ellipsis_type is not None and t.type == ellipsis_type:
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
    ) -> tuple[
        dict[str, tuple[str | None, PyIRNode | None]],
        str | None,
        str | None,
    ]:
        signature: dict[str, tuple[str | None, PyIRNode | None]] = {}
        vararg_name: str | None = None
        kwarg_name: str | None = None

        if not tokens:
            return signature, None, None

        parts = FunctionBuilder._split_top_level_commas(tokens)
        for part in parts:
            part = [t for t in part if not (t.type == tokenize.OP and t.string == ",")]
            if not part:
                raise SyntaxError("Пустой параметр в сигнатуре функции")

            if len(part) == 1 and part[0].type == tokenize.OP and part[0].string in {
                "/",
                "*",
            }:
                # Поддерживаем posonly/kwonly маркеры синтаксически.
                # В py2glua они не вводят дополнительных ограничений.
                continue

            first = part[0]
            if first.type == tokenize.OP and first.string == "*":
                name = FunctionBuilder._parse_star_param(part, is_kwargs=False)
                if vararg_name is not None:
                    FunctionBuilder._raise(
                        "Параметр *args можно объявить только один раз",
                        first,
                    )
                if name in signature or name == kwarg_name:
                    FunctionBuilder._raise(
                        f"Дублирующееся имя параметра: {name!r}",
                        first,
                    )
                vararg_name = name
                continue

            if first.type == tokenize.OP and first.string == "**":
                name = FunctionBuilder._parse_star_param(part, is_kwargs=True)
                if kwarg_name is not None:
                    FunctionBuilder._raise(
                        "Параметр **kwargs можно объявить только один раз",
                        first,
                    )
                if name in signature or name == vararg_name:
                    FunctionBuilder._raise(
                        f"Дублирующееся имя параметра: {name!r}",
                        first,
                    )
                kwarg_name = name
                continue

            name, ann_tokens, default_tokens = FunctionBuilder._parse_single_param(part)

            if name in signature or name == vararg_name or name == kwarg_name:
                FunctionBuilder._raise(
                    f"Дублирующееся имя параметра: {name!r}",
                    part[0],
                )

            ann_str = (
                FunctionBuilder._tokens_to_clean_src(ann_tokens).strip()
                if ann_tokens
                else None
            )

            default_expr: PyIRNode | None = None
            if default_tokens is not None:
                stream = TokenStream(default_tokens)
                default_expr = StatementBuilder._parse_expression(stream)
                if not stream.eof():
                    FunctionBuilder._raise(
                        "Некорректное выражение значения по умолчанию для параметра",
                        stream.peek(),
                    )

            signature[name] = (ann_str, default_expr)

        return signature, vararg_name, kwarg_name

    @staticmethod
    def _parse_star_param(
        tokens: List[tokenize.TokenInfo], *, is_kwargs: bool
    ) -> str:
        first = tokens[0]
        marker = "**" if is_kwargs else "*"
        if first.type != tokenize.OP or first.string != marker:
            FunctionBuilder._raise(
                f"Ожидался маркер '{marker}' в параметре функции",
                first,
            )

        if len(tokens) < 2 or tokens[1].type != tokenize.NAME:
            FunctionBuilder._raise(
                f"Ожидалось имя параметра после '{marker}'",
                tokens[1] if len(tokens) > 1 else first,
            )

        name = tokens[1].string
        rest = tokens[2:]
        if not rest:
            return name

        if not (rest[0].type == tokenize.OP and rest[0].string == ":"):
            FunctionBuilder._raise(
                f"После имени параметра '{marker}{name}' допускается только аннотация типа",
                rest[0],
            )

        ann_tokens = rest[1:]
        if not ann_tokens:
            FunctionBuilder._raise(
                f"Отсутствует аннотация после '{marker}{name}:'",
                rest[0],
            )

        for tok in ann_tokens:
            if tok.type == tokenize.OP and tok.string == "=":
                FunctionBuilder._raise(
                    f"Параметр '{marker}{name}' не может иметь значение по умолчанию",
                    tok,
                )

        return name

    @staticmethod
    def _parse_single_param(
        tokens: List[tokenize.TokenInfo],
    ) -> tuple[str, List[tokenize.TokenInfo], List[tokenize.TokenInfo] | None]:
        if tokens[0].type == tokenize.OP and tokens[0].string in {"*", "**"}:
            FunctionBuilder._raise(
                "Ожидалось имя параметра, а не маркер распаковки",
                tokens[0],
            )

        if tokens[0].type != tokenize.NAME:
            FunctionBuilder._raise("Ожидалось имя параметра", tokens[0])

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
                FunctionBuilder._raise(
                    "Отсутствует значение по умолчанию у параметра",
                    rest[eq_idx],
                )
            default_tokens = rhs
            return name, ann_tokens, default_tokens

        if colon_idx is not None and eq_idx is None:
            ann = rest[colon_idx + 1 :]
            if not ann:
                FunctionBuilder._raise(
                    "Отсутствует аннотация после ':' у параметра",
                    rest[colon_idx],
                )

            ann_tokens = ann
            return name, ann_tokens, None

        if colon_idx is not None and eq_idx is not None:
            if eq_idx < colon_idx:
                FunctionBuilder._raise(
                    "Некорректный синтаксис параметра: '=' расположен до ':'",
                    rest[eq_idx],
                )

            ann = rest[colon_idx + 1 : eq_idx]
            if not ann:
                FunctionBuilder._raise(
                    "Отсутствует аннотация после ':' у параметра",
                    rest[colon_idx],
                )

            rhs = rest[eq_idx + 1 :]
            if not rhs:
                FunctionBuilder._raise(
                    "Отсутствует значение по умолчанию у параметра",
                    rest[eq_idx],
                )

            ann_tokens = ann
            default_tokens = rhs
            return name, ann_tokens, default_tokens

        FunctionBuilder._raise("Некорректный синтаксис параметра", tokens[0])

    # endregion

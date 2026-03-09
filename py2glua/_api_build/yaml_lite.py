from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .._cli import CompilerExit


@dataclass(slots=True)
class _YamlToken:
    line: int
    indent: int
    text: str


def parse_yaml(text: str, *, source_name: str) -> dict[str, Any]:
    tokens = _tokenize_yaml(text, source_name=source_name)
    if not tokens:
        return {}

    first = tokens[0]
    if first.indent != 0:
        _yaml_error(
            source_name,
            f"первая значимая строка должна быть без отступа (line {first.line})",
        )

    node, idx = _parse_yaml_node(tokens, 0, 0, source_name=source_name)
    if idx != len(tokens):
        _yaml_error(source_name, f"не удалось распарсить строку {tokens[idx].line}")

    if not isinstance(node, dict):
        _yaml_error(source_name, "корень должен быть mapping")

    return node


def _yaml_error(source_name: str, msg: str) -> None:
    CompilerExit.user_error(
        f"{source_name}: {msg}",
        show_path=False,
        show_pos=False,
    )


def _tokenize_yaml(text: str, *, source_name: str) -> list[_YamlToken]:
    out: list[_YamlToken] = []

    for ln, raw in enumerate(text.replace("\ufeff", "").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        if "\t" in raw:
            _yaml_error(source_name, f"табы не поддерживаются (line {ln})")

        indent = len(raw) - len(raw.lstrip(" "))
        if indent % 2 != 0:
            _yaml_error(source_name, f"отступ должен быть кратен 2 (line {ln})")

        out.append(_YamlToken(line=ln, indent=indent, text=raw[indent:]))

    return out


def _parse_yaml_node(
    tokens: list[_YamlToken],
    idx: int,
    indent: int,
    *,
    source_name: str,
) -> tuple[Any, int]:
    tok = tokens[idx]
    if tok.indent != indent:
        _yaml_error(source_name, f"неожиданный отступ на line {tok.line}")

    if tok.text.startswith("- "):
        return _parse_yaml_list(tokens, idx, indent, source_name=source_name)

    return _parse_yaml_map(tokens, idx, indent, source_name=source_name)


def _parse_yaml_map(
    tokens: list[_YamlToken],
    idx: int,
    indent: int,
    *,
    source_name: str,
) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}

    while idx < len(tokens):
        tok = tokens[idx]

        if tok.indent < indent:
            break

        if tok.indent > indent:
            _yaml_error(source_name, f"лишний отступ на line {tok.line}")

        if tok.text.startswith("- "):
            break

        if ":" not in tok.text:
            _yaml_error(source_name, f"ожидается 'key: value' на line {tok.line}")

        key, rest = tok.text.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        if not key:
            _yaml_error(source_name, f"пустой ключ на line {tok.line}")

        if key in out:
            _yaml_error(source_name, f"дублирующийся ключ '{key}' (line {tok.line})")

        idx += 1

        if rest:
            out[key] = _parse_scalar(rest)
            continue

        if idx >= len(tokens) or tokens[idx].indent <= indent:
            out[key] = None
            continue

        child, idx = _parse_yaml_node(tokens, idx, tokens[idx].indent, source_name=source_name)
        out[key] = child

    return out, idx


def _parse_yaml_list(
    tokens: list[_YamlToken],
    idx: int,
    indent: int,
    *,
    source_name: str,
) -> tuple[list[Any], int]:
    out: list[Any] = []

    while idx < len(tokens):
        tok = tokens[idx]

        if tok.indent < indent:
            break

        if tok.indent > indent:
            _yaml_error(source_name, f"лишний отступ в списке на line {tok.line}")

        if not tok.text.startswith("- "):
            break

        item_text = tok.text[2:].strip()
        idx += 1

        if not item_text:
            if idx >= len(tokens) or tokens[idx].indent <= indent:
                out.append(None)
                continue

            child, idx = _parse_yaml_node(tokens, idx, tokens[idx].indent, source_name=source_name)
            out.append(child)
            continue

        if _looks_like_map_item(item_text):
            key, rest = item_text.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            item: dict[str, Any] = {}

            if rest:
                item[key] = _parse_scalar(rest)
            else:
                if idx >= len(tokens) or tokens[idx].indent <= indent:
                    item[key] = None
                else:
                    child, idx = _parse_yaml_node(
                        tokens,
                        idx,
                        tokens[idx].indent,
                        source_name=source_name,
                    )
                    item[key] = child

            if idx < len(tokens) and tokens[idx].indent > indent:
                child, idx = _parse_yaml_node(tokens, idx, tokens[idx].indent, source_name=source_name)
                if not isinstance(child, dict):
                    _yaml_error(
                        source_name,
                        f"ожидается mapping continuation для list item (line {tok.line})",
                    )
                for k, v in child.items():
                    if k in item:
                        _yaml_error(
                            source_name,
                            f"duplicate key '{k}' в list item (line {tok.line})",
                        )
                    item[k] = v

            out.append(item)
            continue

        out.append(_parse_scalar(item_text))

        if idx < len(tokens) and tokens[idx].indent > indent:
            _yaml_error(
                source_name,
                f"scalar list item не может иметь вложенность (line {tok.line})",
            )

    return out, idx


def _parse_scalar(text: str) -> Any:
    s = text.strip()

    if (len(s) >= 2 and s[0] == '"' and s[-1] == '"') or (
        len(s) >= 2 and s[0] == "'" and s[-1] == "'"
    ):
        return s[1:-1]

    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none", "~"}:
        return None

    if re.match(r"^-?[0-9]+$", s):
        try:
            return int(s)
        except Exception:
            pass

    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []

        return [_parse_scalar(part) for part in _split_inline_csv(inner)]

    return s


def _split_inline_csv(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []

    quote: str | None = None
    depth = 0

    for ch in text:
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue

        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            continue

        if ch == "[":
            depth += 1
            buf.append(ch)
            continue

        if ch == "]":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue

        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    return parts


def _looks_like_map_item(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_\-]*\s*:", text))

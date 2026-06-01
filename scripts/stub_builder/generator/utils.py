from textwrap import dedent, indent


def format_docstring(text: str, base_indent: int = 4) -> str:
    if not text:
        return ""

    clean = dedent(text).strip().replace('"""', '\\"\\"\\"')
    body = indent(clean, " " * base_indent)

    return f'"""{body}\n{" " * base_indent}"""'

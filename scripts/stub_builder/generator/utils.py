def format_docstring(text: str, base_indent: int = 4) -> str:
    if not text:
        return ""

    text = text.expandtabs(4)
    lines = text.splitlines()
    min_indent = float("inf")
    for line in lines:
        if line.strip():
            spaces = len(line) - len(line.lstrip())
            if spaces < min_indent:
                min_indent = spaces

    if min_indent != float("inf") and min_indent > 0:
        lines = [line[min_indent:] if line.strip() else line.strip() for line in lines]

    clean = "\n".join(lines).strip()
    clean = clean.replace('"""', '\\"\\"\\"')
    ind = " " * base_indent
    body_lines = []
    for line in clean.splitlines():
        if line.strip():
            body_lines.append(f"{ind}{line}")

        else:
            body_lines.append("")

    body = "\n".join(body_lines)

    return f'{ind}"""\n{body}\n{ind}"""'

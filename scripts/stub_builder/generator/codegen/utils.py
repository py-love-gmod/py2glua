import keyword
import re
import textwrap


def sanitize_name(name: str) -> str:
    """
    Превращает произвольную строку в корректный Python идентификатор.
    - Заменяет пробелы и недопустимые символы на '_'
    - Схлопывает множественные подчёркивания
    - Убирает начальные/конечные подчёркивания, кроме случаев ключевых слов
    - Добавляет '_' в начало, если имя начинается с цифры
    - Применяет escape_keyword для ключевых слов
    """
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = "_" + name

    if not name:
        name = "arg"

    return escape_keyword(name)


def escape_keyword(name: str) -> str:
    if keyword.iskeyword(name) or name in {"type", "match", "case"}:
        return f"{name}_"

    return name


def format_docstring(text: str, base_indent: int = 4) -> str:
    if not text:
        return ""

    text = textwrap.dedent(text).strip()
    text = text.replace('"""', '\\"\\"\\"')
    ind = " " * base_indent
    body_lines: list[str] = []
    for line in text.splitlines():
        if line.strip():
            body_lines.append(f"{ind}{line}")

        else:
            body_lines.append("")

    body = "\n".join(body_lines)
    return f'{ind}"""\n{body}\n{ind}"""'

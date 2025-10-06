import ast

from ..exceptions import NameError


class NameGuard(ast.NodeVisitor):
    LUA_KEYWORDS = {
        "and",
        "break",
        "do",
        "else",
        "elseif",
        "end",
        "false",
        "for",
        "function",
        "goto",
        "if",
        "in",
        "local",
        "nil",
        "not",
        "or",
        "repeat",
        "return",
        "then",
        "true",
        "until",
        "while",
    }

    SYSTEM_GLOBALS = {
        "string",
        "table",
        "math",
        "os",
        "io",
        "file",
        "debug",
        "coroutine",
        "package",
        "jit",
    }

    DEFAULT_FORBIDDEN = LUA_KEYWORDS | SYSTEM_GLOBALS

    def __init__(self, forbidden: set[str] | None = None) -> None:
        self.forbidden = set(forbidden or self.DEFAULT_FORBIDDEN)
        self.violations: list[tuple[int, str]] = []

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.forbidden:
            self.violations.append((node.lineno, node.id))

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in self.forbidden:
            self.violations.append((node.lineno, node.name))

        for arg in node.args.args:
            if arg.arg in self.forbidden:
                self.violations.append((arg.lineno, arg.arg))

        self.generic_visit(node)

    def validate(self, tree: ast.AST, filename: str | None = None) -> None:
        self.visit(tree)
        if self.violations:
            msgs = [
                f"Line {ln}: forbidden name '{name}'" for ln, name in self.violations
            ]
            filepart = f"{filename or '<unknown>'}"
            raise NameError(f"{filepart}:\n" + "\n".join(msgs))

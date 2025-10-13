import ast
import logging
from pathlib import Path
from typing import NamedTuple

from ..file_meta import Priority
from ..runtime import Realm

logger = logging.getLogger("py2glua")


class _MagicIntResult(NamedTuple):
    value: int | ast.Attribute
    lineno: int
    col_offset: int


class FileCTX:
    def __init__(self) -> None:
        # base file data
        self.file_path: Path

        # ast info
        self.file_ast: ast.Module

        # analyze
        self.file_realm: Realm
        self.file_priority: int
        self.file_globals: set[str] = set()  # TODO:
        self.file_imports: dict[str, str] = {}  # TODO:

    def build(self, path: Path) -> None:
        if not self._load_ast(path):
            return

        self._resolve_imports()
        self._remove_enter_point()
        self.file_realm = self._get_realm()
        self.file_priority = self._get_priority()
        self._cleanup_magic_ints(("__realm___", "__priority__"))

    def _load_ast(self, path: Path) -> bool:
        if not path.exists():
            logger.error(f"[FileCTX build] file dont exists\nPath: {path}")
            return False

        if path.suffix.lower() != ".py":
            logger.error(f"[FileCTX build] not a .py file\nPath: {path}")
            return False

        try:
            self.file_ast = ast.parse(path.read_text(encoding="utf-8-sig"))
            self.file_path = path
            return True

        except Exception as err:
            logger.error(f"[FileCTX build] {err}")

        return False

    # region cleanup file
    def _resolve_imports(self) -> None:
        """
        `import a.b as c` -> `import a`

        `from x.y import z as w` -> `import x`

        Все обращения к локальным алиасам (`c`, `w`) заменяем на полную цепочку

        `c.foo` -> `a.b.foo`

        `w()` -> `x.y.z()`
        """
        if self.file_ast is None:
            return

        local_to_parts: dict[str, list[str]] = {}

        class ImportNormalizer(ast.NodeTransformer):
            def __init__(self, alias_map: dict[str, list[str]]):
                self.alias_map = alias_map
                super().__init__()

            def visit_Import(self, node: ast.Import):
                new_names: list[ast.alias] = []
                for alias in node.names:
                    parts = alias.name.split(".")
                    local_name = alias.asname or parts[0]
                    self.alias_map[local_name] = parts[:]
                    top = parts[0]
                    if not any(n.name == top for n in new_names):
                        new_names.append(ast.alias(name=top, asname=None))

                node.names = new_names
                return node

            def visit_ImportFrom(self, node: ast.ImportFrom):
                module = node.module or ""
                module_parts = module.split(".") if module else []
                new_top = (module_parts[0] if module_parts else None) or node.names[
                    0
                ].name.split(".")[0]

                for alias in node.names:
                    local_name = alias.asname or alias.name
                    full_parts = (
                        module_parts + [alias.name] if module_parts else [alias.name]
                    )
                    self.alias_map[local_name] = full_parts

                node_repl = ast.Import(names=[ast.alias(name=new_top, asname=None)])
                return node_repl

            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load) and node.id in self.alias_map:
                    parts = self.alias_map[node.id]
                    if len(parts) == 1:
                        return ast.copy_location(
                            ast.Name(id=parts[0], ctx=node.ctx), node
                        )

                    value: ast.AST = ast.copy_location(
                        ast.Name(id=parts[0], ctx=node.ctx), node
                    )
                    for attr in parts[1:]:
                        value = ast.copy_location(
                            ast.Attribute(value=value, attr=attr, ctx=node.ctx), node
                        )

                    return value

                return node

        normalizer = ImportNormalizer(local_to_parts)
        self.file_ast = normalizer.visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

        class NameRewriter(ast.NodeTransformer):
            def __init__(self, alias_map: dict[str, list[str]]):
                self.alias_map = alias_map
                super().__init__()

            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load) and node.id in self.alias_map:
                    parts = self.alias_map[node.id]
                    if len(parts) == 1:
                        return ast.copy_location(
                            ast.Name(id=parts[0], ctx=node.ctx), node
                        )

                    value: ast.AST = ast.copy_location(
                        ast.Name(id=parts[0], ctx=node.ctx), node
                    )

                    for attr in parts[1:]:
                        value = ast.copy_location(
                            ast.Attribute(value=value, attr=attr, ctx=node.ctx), node
                        )

                    return value

                return node

        rewriter = NameRewriter(local_to_parts)
        self.file_ast = rewriter.visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

    def _remove_enter_point(self) -> None:
        class MainBlockRemover(ast.NodeTransformer):
            def visit_If(self, node: ast.If):
                if (
                    isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"
                ):
                    return None

                return node

        self.file_ast = MainBlockRemover().visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

    def _cleanup_magic_ints(self, names: tuple[str, ...]) -> None:
        class MagicRemover(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign):
                new_targets = [
                    t
                    for t in node.targets
                    if not (isinstance(t, ast.Name) and t.id in names)
                ]
                if not new_targets:
                    return None

                node.targets = new_targets
                return node

        self.file_ast = MagicRemover().visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

    # endregion

    # region magic int for file
    def _extract_magic_int(self, name: str) -> _MagicIntResult | None:
        for node in ast.walk(self.file_ast):
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if not isinstance(target, ast.Name) or target.id != name:
                    continue

                if isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, int
                ):
                    return _MagicIntResult(
                        value=node.value.value,
                        lineno=node.value.lineno,
                        col_offset=node.value.col_offset,
                    )

                elif isinstance(node.value, ast.Attribute):
                    return _MagicIntResult(
                        value=node.value,
                        lineno=node.value.lineno,
                        col_offset=node.value.col_offset,
                    )

        return None

    def _get_realm(self) -> Realm:
        res = self._extract_magic_int("__realm__")
        if res is None:
            logger.warning(
                f"[FileCTX build] empty __realm__ value for file. Set to Realm.Server\nPATH: {self.file_path}"
            )
            return Realm.SERVER

        val = res.value if isinstance(res.value, int) else 0
        if val < 1 or val > 3:
            logger.error(
                f"[FileCTX build] invalid __realm__. Set to Realm.Server\nPATH: {self.file_path}\nLINE|OFFSET: {res.lineno}|{res.col_offset}"
            )
            return Realm.SERVER

        return Realm(val)

    def _get_priority(self) -> int:
        res = self._extract_magic_int("__priority__")
        if res is None:
            return Priority.MEDIUM.value

        val = res.value if isinstance(res.value, int) else Priority.MEDIUM.value
        return int(val)

    # endregion

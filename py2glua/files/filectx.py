import ast
import logging
import sys
import sysconfig
from pathlib import Path
from typing import Literal, NamedTuple

from ..file_meta import Priority
from ..runtime import Realm

logger = logging.getLogger("py2glua")

# region import analyze
_STDLIB_PATH = Path(sysconfig.get_path("stdlib"))
_SITE_PACKAGES = [
    Path(p) for p in sys.path if "site-packages" in p or "dist-packages" in p
]


def _classify_import(
    name: str,
    project_root: Path,
) -> Literal["std", "loc", "ext", "unk"]:
    if name in sys.builtin_module_names:
        return "std"

    if (project_root / f"{name}.py").exists() or (project_root / name).is_dir():
        return "loc"

    for sp in _SITE_PACKAGES:
        if (sp / name).exists() or (sp / f"{name}.py").exists():
            return "ext"

    if (_STDLIB_PATH / f"{name}.py").exists() or (_STDLIB_PATH / name).is_dir():
        return "std"

    return "unk"


def _package_parts_from_file(file_path: Path) -> list[str]:
    parts: list[str] = []
    p = file_path.parent
    while True:
        init = p / "__init__.py"
        if init.exists():
            parts.insert(0, p.name)
            p = p.parent
            if p == p.parent:
                break

        else:
            break

    return parts


class _ImportNormalizer(ast.NodeTransformer):
    def __init__(self, alias_map: dict[str, list[str]], file_path: Path | None):
        self.alias_map = alias_map
        self.file_path = Path(file_path) if file_path is not None else None
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
        if node.level and node.level > 0:
            if self.file_path is None:
                top = node.names[0].name.split(".")[0]
                return ast.Import(names=[ast.alias(name=top, asname=None)])

            package_parts = _package_parts_from_file(self.file_path)
            base_len = len(package_parts) - (node.level - 1)
            if base_len < 0:
                base = []
            else:
                base = package_parts[:base_len]

            module_parts = node.module.split(".") if node.module else []
            for alias in node.names:
                alias_name_parts = alias.name.split(".")
                full_parts = base + module_parts + alias_name_parts
                local_name = alias.asname or alias.name
                self.alias_map[local_name] = full_parts

            if self.alias_map:
                first_full = next(iter(self.alias_map.values()))
                top = first_full[0] if first_full else node.names[0].name.split(".")[0]
                return ast.Import(names=[ast.alias(name=top, asname=None)])

            else:
                top = node.names[0].name.split(".")[0]
                return ast.Import(names=[ast.alias(name=top, asname=None)])

        else:
            module = node.module or ""
            module_parts = module.split(".") if module else []
            new_top = (module_parts[0] if module_parts else None) or node.names[
                0
            ].name.split(".")[0]

            for alias in node.names:
                local_name = alias.asname or alias.name
                full_parts = (
                    (module_parts + [alias.name]) if module_parts else [alias.name]
                )
                self.alias_map[local_name] = full_parts

            node_repl = ast.Import(names=[ast.alias(name=new_top, asname=None)])
            return node_repl


class _NameRewriter(ast.NodeTransformer):
    def __init__(self, alias_map: dict[str, list[str]]):
        self.alias_map = alias_map
        super().__init__()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.alias_map:
            parts = self.alias_map[node.id]
            if len(parts) == 1:
                return ast.copy_location(ast.Name(id=parts[0], ctx=node.ctx), node)

            value: ast.AST = ast.copy_location(
                ast.Name(id=parts[0], ctx=node.ctx), node
            )

            for attr in parts[1:]:
                value = ast.copy_location(
                    ast.Attribute(value=value, attr=attr, ctx=node.ctx), node
                )

            return value

        return node


# endregion


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
        self.file_imports: set[str] = set()

    def build(self, path: Path, source_path: Path) -> None:
        if not self._load_ast(path):
            return

        self._resolve_imports()
        self._build_file_imports(source_path)
        self.file_realm = self._get_realm()
        self.file_priority = self._get_priority()
        self._cleanup_magic_ints(("__realm___", "__priority__"))
        self._remove_enter_point()

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

    # region imports
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
        normalizer = _ImportNormalizer(local_to_parts, file_path=self.file_path)
        self.file_ast = normalizer.visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

        rewriter = _NameRewriter(local_to_parts)
        self.file_ast = rewriter.visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

    def _build_file_imports(self, source_path) -> None:
        imports: set[str] = set()

        for node in ast.walk(self.file_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    full_name = alias.name
                    top_module = full_name.split(".")[0]
                    kind = _classify_import(top_module, source_path)
                    imports.add(f"{kind}|{full_name}")

        self.file_imports = imports

    # endregion

    # region cleanup file

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

import ast
import logging
import sys
import sysconfig
from pathlib import Path
from typing import Literal, NamedTuple

from ..exceptions import CompileError
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
    value: int
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
        self.file_globals: set[str] = set()
        self.file_imports: set[str] = set()

    def build(self, path: Path, source_path: Path) -> None:
        self._load_ast(path)
        self._resolve_imports()
        self._build_file_imports(source_path)
        self.file_realm = self._get_realm()
        self.file_priority = self._get_priority()
        self._collect_globals()
        self._remove_enter_point()

    def _load_ast(self, path: Path) -> None:
        if not path.exists():
            raise CompileError("[FileCTX build] file dont exists", path, 0, 0)

        if path.suffix.lower() != ".py":
            raise CompileError("[FileCTX build] not a .py file", path, 0, 0)

        try:
            self.file_ast = ast.parse(path.read_text(encoding="utf-8-sig"))
            self.file_path = path

        except Exception as err:
            raise CompileError(f"[FileCTX build] {err}", path, 0, 0)

    # region imports
    def _resolve_imports(self) -> None:
        if self.file_ast is None:
            return

        local_to_parts: dict[str, list[str]] = {}
        normalizer = _ImportNormalizer(local_to_parts, file_path=self.file_path)
        self.file_ast = normalizer.visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

        rewriter = _NameRewriter(local_to_parts)
        self.file_ast = rewriter.visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

    def _build_file_imports(self, source_path: Path) -> None:
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

    def _cleanup_magic_int(self, name: str) -> None:
        class MagicRemover(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign):
                new_targets = [
                    t
                    for t in node.targets
                    if not (isinstance(t, ast.Name) and t.id == name)
                ]
                if not new_targets:
                    return None

                node.targets = new_targets
                return node

        self.file_ast = MagicRemover().visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

    # endregion

    # region magic int for file
    def _get_full_attr_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            parent = self._get_full_attr_name(node.value)
            if parent is None:
                return None

            return f"{parent}.{node.attr}"

        return None

    def _resolve_enum_attr_to_int(self, node: ast.Attribute) -> int | None:
        full = self._get_full_attr_name(node)
        if not full:
            return None

        parts = full.split(".")
        if len(parts) < 2:
            return None

        root, attr = parts[-2], parts[-1]

        if root == "Realm":
            try:
                return getattr(Realm, attr).value

            except AttributeError:
                return None

        if root == "Priority":
            try:
                return getattr(Priority, attr).value

            except AttributeError:
                return None

        return None

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

                if isinstance(node.value, ast.Attribute):
                    resolved = self._resolve_enum_attr_to_int(node.value)
                    if resolved is not None:
                        return _MagicIntResult(
                            value=resolved,
                            lineno=node.value.lineno,
                            col_offset=node.value.col_offset,
                        )

        return None

    def _get_realm(self) -> Realm:
        res = self._extract_magic_int("__realm__")
        if res is None:
            logger.warning(
                f"[FileCTX build] empty __realm__ value for file. Set to Realm.SERVER\nPATH: {self.file_path}"
            )
            return Realm.SERVER

        val = res.value
        if val < 1 or val > 3:
            raise CompileError(
                "[FileCTX build] invalid __realm__",
                self.file_path,
                res.lineno,
                res.col_offset,
            )

        self._cleanup_magic_int("__realm__")
        return Realm(val)

    def _get_priority(self) -> int:
        res = self._extract_magic_int("__priority__")
        if res is None:
            return Priority.MEDIUM.value

        val = res.value
        try:
            ival = int(val)

        except Exception:
            ival = Priority.MEDIUM.value

        self._cleanup_magic_int("__priority__")
        return ival

    # endregion

    # region globals

    def _collect_globals(self) -> None:
        self.file_globals = set()

        def get_full_attr_name(node: ast.AST) -> str | None:
            return self._get_full_attr_name(node)

        class GlobalCollector(ast.NodeTransformer):
            def __init__(self, outer: "FileCTX"):
                self.outer = outer
                self.declared: set[str] = set()
                super().__init__()

            def _handle_global_var(self, node: ast.Call, targets: list[ast.Name]):
                if len(node.args) != 1:
                    raise CompileError(
                        f"[FileCTX build] Global.var() called with {len(node.args)} argument(s).",
                        self.outer.file_path,
                        node.lineno,
                        node.col_offset,
                    )

                for t in targets:
                    self.outer.file_globals.add(f"var|{t.id}")

            def visit_Assign(self, node: ast.Assign):
                name_targets = [t for t in node.targets if isinstance(t, ast.Name)]

                if isinstance(node.value, ast.Call):
                    func_name = get_full_attr_name(node.value.func)
                    if func_name and func_name.endswith("Global.var"):
                        if not name_targets:
                            raise CompileError(
                                "[FileCTX build] Global.var used in complex assignment",
                                self.outer.file_path,
                                node.lineno,
                                node.col_offset,
                            )

                        else:
                            already = [
                                t.id for t in name_targets if t.id in self.declared
                            ]
                            if already:
                                raise CompileError(
                                    f"[FileCTX build] Global.var called for already declared variable: {already}",
                                    self.outer.file_path,
                                    node.lineno,
                                    node.col_offset,
                                )

                            else:
                                self._handle_global_var(node.value, name_targets)
                                if len(node.value.args) == 1 and isinstance(
                                    node.value.args[0], ast.Constant
                                ):
                                    node.value = node.value.args[0]

                for t in name_targets:
                    self.declared.add(t.id)

                return node

            def visit_AnnAssign(self, node: ast.AnnAssign):
                target_name = (
                    node.target.id if isinstance(node.target, ast.Name) else None
                )

                if isinstance(node.value, ast.Call):
                    func_name = get_full_attr_name(node.value.func)
                    if func_name and func_name.endswith("Global.var"):
                        if target_name is None:
                            raise CompileError(
                                "[FileCTX build] Global.var used in annotated complex target",
                                self.outer.file_path,
                                node.lineno,
                                node.col_offset,
                            )

                        else:
                            if target_name in self.declared:
                                raise CompileError(
                                    f"[FileCTX build] Global.var called for already declared variable: {target_name}",
                                    self.outer.file_path,
                                    node.lineno,
                                    node.col_offset,
                                )

                            else:
                                fake_name_node = ast.Name(
                                    id=target_name, ctx=ast.Store()
                                )
                                self._handle_global_var(node.value, [fake_name_node])
                                if len(node.value.args) == 1 and isinstance(
                                    node.value.args[0], ast.Constant
                                ):
                                    node.value = node.value.args[0]

                if target_name:
                    self.declared.add(target_name)

                return node

            def visit_FunctionDef(self, node: ast.FunctionDef):
                new_decorators = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call):
                        func_name = get_full_attr_name(dec.func)
                        if func_name and func_name.endswith("Global.mark"):
                            self.outer.file_globals.add(f"func|{node.name}")
                            continue

                    new_decorators.append(dec)

                node.decorator_list = new_decorators

                self.declared.add(node.name)
                return node

            def visit_ClassDef(self, node: ast.ClassDef):
                new_decorators = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call):
                        func_name = get_full_attr_name(dec.func)
                        if func_name and func_name.endswith("Global.mark"):
                            self.outer.file_globals.add(f"class|{node.name}")
                            continue

                    new_decorators.append(dec)

                node.decorator_list = new_decorators

                self.declared.add(node.name)
                return node

        self.file_ast = GlobalCollector(self).visit(self.file_ast)
        ast.fix_missing_locations(self.file_ast)

        # endregion

import importlib.util
import sys
import sysconfig
from pathlib import Path
from typing import Sequence

from ...config import Py2GluaConfig
from ..py_logic_block_builder import PyLogicNode
from .py_ir_dataclass import PyIRContext, PyIRImport, PyIRImportType, PyIRNode


class ImportAnalyzer:
    @staticmethod
    def build(
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> tuple[int, Sequence[PyIRNode]]:
        origin = node.origins[0]
        if origin is None:
            raise ValueError("Origin is None")

        context: PyIRContext | None = getattr(parent_obj, "context", None)
        if context is None:
            raise ValueError("parent_obj context is missing")

        tokens = [t.string for t in origin.tokens if hasattr(t, "string")]
        if not tokens:
            return (1, [])

        modules: list[str] = []

        if tokens[0] == "import":
            i = 1
            while i < len(tokens):
                t = tokens[i]
                if t in {",", "\n"}:
                    i += 1
                    continue

                if t == "as" and i + 1 < len(tokens):
                    alias = tokens[i + 1]
                    real_name = modules[-1]
                    context.meta["aliases"][alias] = real_name
                    i += 2
                    continue

                modules.append(t)
                i += 1

        elif tokens[0] == "from":
            try:
                imp_idx = tokens.index("import")

            except ValueError:
                raise SyntaxError("Malformed 'from' import statement")

            pkg = ".".join(t for t in tokens[1:imp_idx] if t not in {".", "\n"}).strip()
            if not pkg:
                raise SyntaxError("Empty module path in 'from' import")

            names = tokens[imp_idx + 1 :]
            if "*" in names:
                raise SyntaxError("Wildcard import (*) is forbidden")

            i = 0
            while i < len(names):
                t = names[i]
                if t in {",", "\n"}:
                    i += 1
                    continue

                if t == "as" and i + 1 < len(names):
                    alias = names[i + 1]
                    real_name = f"{pkg}.{names[i - 1]}"
                    context.meta["aliases"][alias] = real_name
                    i += 2
                    continue

                modules.append(f"{pkg}.{t}")
                i += 1

        else:
            return (1, [])

        project_root = Py2GluaConfig.source.resolve()
        std_paths = sysconfig.get_paths()
        std_base = std_paths.get("stdlib") or sys.base_prefix
        stdlib_dir = Path(std_base).resolve()

        ir_nodes: list[PyIRImport] = []
        for mod in modules:
            i_type = ImportAnalyzer.resolve_type(mod, project_root, stdlib_dir)
            if i_type == PyIRImportType.EXTERNAL:
                raise RuntimeError(
                    f"External dependency '{mod}' detected. External pip packages are not supported."
                )

            ir_node = PyIRImport(
                line=origin.tokens[0].start[0],
                offset=origin.tokens[0].start[1],
                modules=[mod],
                i_type=i_type,
            )
            context.meta["imports"].append(mod)
            ir_nodes.append(ir_node)

        return (1, ir_nodes)

    @staticmethod
    def resolve_type(
        module: str, project_root: Path, stdlib_dir: Path
    ) -> PyIRImportType:
        if not module:
            return PyIRImportType.UNKNOWN

        if module.startswith("."):
            return PyIRImportType.LOCAL

        try:
            spec = importlib.util.find_spec(module)

        except Exception:
            spec = None

        if spec and getattr(spec, "origin", None) in {"built-in", "frozen"}:
            return PyIRImportType.STD_LIB

        if not spec or not getattr(spec, "origin", None):
            if module in sys.builtin_module_names:
                return PyIRImportType.STD_LIB

            return PyIRImportType.UNKNOWN

        try:
            mod_path = Path(spec.origin).resolve()  # type: ignore[arg-type]

        except Exception:
            return PyIRImportType.UNKNOWN

        project_root = project_root.resolve()
        stdlib_dir = stdlib_dir.resolve()

        if stdlib_dir in mod_path.parents or mod_path == stdlib_dir:
            return PyIRImportType.STD_LIB

        if project_root in mod_path.parents or mod_path == project_root:
            return PyIRImportType.LOCAL

        purelib = Path(sysconfig.get_paths().get("purelib", "") or ".").resolve()
        platlib = Path(sysconfig.get_paths().get("platlib", "") or ".").resolve()
        if purelib in mod_path.parents or platlib in mod_path.parents:
            return PyIRImportType.EXTERNAL

        return PyIRImportType.EXTERNAL

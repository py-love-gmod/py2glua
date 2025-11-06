import importlib.util
import sys
import sysconfig
import tokenize
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Any

from ..config import Py2GluaConfig
from .py_logic_block_builder import (
    PublicLogicKind,
    PublicLogicNode,
    PyLogicBlockBuilder,
)


# region IR Nodes
@dataclass
class PyIRNode:
    line: int | None
    offset: int | None

    def walk(self):
        raise NotImplementedError()


@dataclass
class PyIRFile(PyIRNode):
    path: Path
    body: list[PyIRNode] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def walk(self):
        yield self
        for node in self.body:
            yield node


# region statement
@dataclass
class PyIRVarCreate(PyIRNode):
    vid: int
    is_global: bool = False
    external: bool = False

    def walk(self):
        yield self


@dataclass
class PyIRVarUse(PyIRNode):
    vid: int

    def walk(self):
        yield self


@dataclass
class PyIRAtt(PyIRNode):
    name: str

    def walk(self):
        yield self


@dataclass
class PyIRCall(PyIRNode):
    name: str
    args: list[object] = field(default_factory=list)
    kwargs: dict[str, object] = field(default_factory=dict)

    def walk(self):
        yield self


@dataclass
class PyIRReturn(PyIRNode):
    value: PyIRNode

    def walk(self):
        yield self
        yield self.value


@dataclass
class PyIRDel(PyIRNode):
    value: PyIRNode

    def walk(self):
        yield self
        yield self.value


@dataclass
class PyIRConstant(PyIRNode):
    value: object

    def walk(self):
        yield self


# endregion


# region import
class PyIRImportType(IntEnum):
    UNKNOWN = auto()
    LOCAL = auto()
    STD_LIB = auto()
    INTERNAL = auto()
    EXTERNAL = auto()


@dataclass
class PyIRImport(PyIRNode):
    modules: list[str]
    i_type: PyIRImportType = PyIRImportType.UNKNOWN

    def walk(self):
        yield self


# endregion

# endregion


# region Builder
class PyIRBuilder:
    @classmethod
    def build(cls, path_to_file: Path) -> PyIRFile:
        logic_blocks = PyLogicBlockBuilder.build(path_to_file)
        py_ir_file = PyIRFile(
            line=None,
            offset=None,
            path=path_to_file,
            meta={
                "imports": [],
                "aliases": {},
                "var_name": {},
            },
        )
        py_ir_file.body = cls._build_ir_block(logic_blocks, py_ir_file)
        return py_ir_file

    # region build
    @classmethod
    def _build_ir_block(
        cls,
        nodes: list[PublicLogicNode],
        file_obj: PyIRFile,
    ) -> list[PyIRNode]:
        dispatch = {
            PublicLogicKind.IMPORT: cls._build_ir_import,
            PublicLogicKind.STATEMENT: cls._build_ir_statement,
        }

        out: list[PyIRNode] = []
        i = 0
        n = len(nodes)

        while i < n:
            node = nodes[i]
            func = dispatch.get(node.kind)
            if func is None:
                raise ValueError(
                    f"PublicLogicKind type {node.kind} no func to build ir"
                )

            offset, result_nodes = func(file_obj, nodes, i)
            out.extend(result_nodes)
            i += offset

        return out

    # endregion

    # region import
    @classmethod
    def _build_ir_import(
        cls,
        file_obj: PyIRFile,
        node_list: list[PublicLogicNode],
        start: int,
    ) -> tuple[int, list[PyIRNode]]:
        node = node_list[start]
        origin = node.origin
        if origin is None:
            raise ValueError("Origin is None")

        tokens = [t.string for t in origin.tokens if hasattr(t, "string")]
        if not tokens:
            return (1, [])

        modules: list[str] = []

        # import X, Y as Z
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
                    file_obj.meta["aliases"][alias] = real_name
                    i += 2
                    continue

                modules.append(t)
                i += 1

        # from X import A, B as C
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
                    file_obj.meta["aliases"][alias] = real_name
                    i += 2
                    continue

                modules.append(f"{pkg}.{t}")
                i += 1

        else:
            return (1, [])

        project_root = Path(Py2GluaConfig.data["source"]).resolve()
        stdlib_dir = Path(sysconfig.get_paths()["stdlib"]).resolve()

        ir_nodes: list[PyIRImport] = []
        for mod in modules:
            i_type = cls._resolve_import_type(mod, project_root, stdlib_dir)
            if i_type == PyIRImportType.EXTERNAL:
                raise RuntimeError(
                    f"External dependency '{mod}' detected. "
                    "External pip packages are not supported."
                )

            ir_node = PyIRImport(
                line=origin.tokens[0].start[0],
                offset=origin.tokens[0].start[1],
                modules=[mod],
                i_type=i_type,
            )
            file_obj.meta["imports"].append(mod)
            ir_nodes.append(ir_node)

        return (1, ir_nodes)  # type: ignore

    @staticmethod
    def _resolve_import_type(
        module: str,
        project_root: Path,
        stdlib_dir: Path,
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
            mod_path = Path(spec.origin).resolve()  # type: ignore

        except Exception:
            return PyIRImportType.UNKNOWN

        project_root = project_root.resolve()
        stdlib_dir = stdlib_dir.resolve()

        # STD_LIB
        if stdlib_dir in mod_path.parents or mod_path == stdlib_dir:
            return PyIRImportType.STD_LIB

        # LOCAL
        if project_root in mod_path.parents or mod_path == project_root:
            return PyIRImportType.LOCAL

        # EXTERNAL
        purelib = Path(sysconfig.get_paths().get("purelib", "") or ".").resolve()
        platlib = Path(sysconfig.get_paths().get("platlib", "") or ".").resolve()
        if purelib in mod_path.parents or platlib in mod_path.parents:
            return PyIRImportType.EXTERNAL

        return PyIRImportType.EXTERNAL

    # endregion

    # region statment
    @classmethod
    def _build_ir_statement(
        cls,
        file_obj: PyIRFile,
        node_list: list[PublicLogicNode],
        start: int,
    ) -> tuple[int, list[PyIRNode]]:
        node = node_list[start]
        origin = node.origin
        if origin is None:
            raise ValueError("Origin is None")

        ir_nodes = ParseStatment.parse(origin.tokens, file_obj)
        return (1, [ir_nodes])

    # endregion


# endregion


# region ParseStatment
class ParseStatment:
    @classmethod
    def parse(
        cls,
        tokens: list,
        file_obj: PyIRFile,
    ) -> PyIRNode:
        if not tokens:
            raise ValueError("OwO, some tokens are empty for some reason")

        first: tokenize.TokenInfo = tokens[0]
        line, offset = first.start

        raise ValueError("OwO, some tokens are leaked")


# endregion

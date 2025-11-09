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

_NO_LIT = object()


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
    body: list["PyIRNode"] = field(default_factory=list)
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
class PyIRMethodCall(PyIRNode):
    obj: PyIRNode
    method: str
    args: list[object] = field(default_factory=list)
    kwargs: dict[str, object] = field(default_factory=dict)

    def walk(self):
        yield self


@dataclass
class PyIRIndex(PyIRNode):
    obj: PyIRNode
    index: PyIRNode

    def walk(self):
        yield self
        yield self.obj
        yield self.index


@dataclass
class PyIRUnaryOp(PyIRNode):
    op: str
    value: PyIRNode

    def walk(self):
        yield self
        yield self.value


@dataclass
class PyIRBinOp(PyIRNode):
    op: str
    left: PyIRNode
    right: PyIRNode

    def walk(self):
        yield self
        yield self.left
        yield self.right


@dataclass
class PyIRAssign(PyIRNode):
    target_vid: int
    value: PyIRNode

    def walk(self):
        yield self
        yield self.value


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


# region lexing
class Tok:
    def __init__(self, toks: list[tokenize.TokenInfo]) -> None:
        layout = {
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.ENDMARKER,
        }
        self.toks = [t for t in toks if t.type not in layout and hasattr(t, "string")]
        self.i = 0

    def peek(self, k: int = 0) -> tokenize.TokenInfo | None:
        j = self.i + k
        return self.toks[j] if 0 <= j < len(self.toks) else None

    def advance(self) -> tokenize.TokenInfo:
        t = self.toks[self.i]
        self.i += 1
        return t

    def accept_op(self, op: str) -> bool:
        t = self.peek()
        if t and t.type == tokenize.OP and t.string == op:
            self.i += 1
            return True

        return False

    def next_is_op(self, op: str) -> bool:
        t = self.peek()
        return bool(t and t.type == tokenize.OP and t.string == op)

    def next_is_name(self, s: str | None = None) -> bool:
        t = self.peek()
        if not t or t.type != tokenize.NAME:
            return False

        return True if s is None else (t.string == s)


# endregion


# region parser_expr
class ExprParser:
    @staticmethod
    def parse(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        return ExprParser._parse_or(ts, file_obj)

    @staticmethod
    def _parse_or(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        left = ExprParser._parse_and(ts, file_obj)
        while ts.next_is_name("or"):
            op = ts.advance()
            right = ExprParser._parse_and(ts, file_obj)
            if right is None or left is None:
                return left

            left = PyIRBinOp(
                line=op.start[0], offset=op.start[1], op="or", left=left, right=right
            )

        return left

    @staticmethod
    def _parse_and(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        left = ExprParser._parse_compare(ts, file_obj)
        while ts.next_is_name("and"):
            op = ts.advance()
            right = ExprParser._parse_compare(ts, file_obj)
            if right is None or left is None:
                return left

            left = PyIRBinOp(
                line=op.start[0], offset=op.start[1], op="and", left=left, right=right
            )

        return left

    @staticmethod
    def _parse_compare(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        left = ExprParser._parse_add(ts, file_obj)
        while True:
            t = ts.peek()
            if (
                not t
                or t.type != tokenize.OP
                or t.string not in {"==", "!=", "<", "<=", ">", ">="}
            ):
                break

            op = ts.advance()
            right = ExprParser._parse_add(ts, file_obj)
            if right is None or left is None:
                return left

            left = PyIRBinOp(
                line=op.start[0],
                offset=op.start[1],
                op=op.string,
                left=left,
                right=right,
            )

        return left

    @staticmethod
    def _parse_add(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        left = ExprParser._parse_mul(ts, file_obj)
        while True:
            t = ts.peek()
            if not t or t.type != tokenize.OP or t.string not in {"+", "-"}:
                break

            op = ts.advance()
            right = ExprParser._parse_mul(ts, file_obj)
            if right is None or left is None:
                return left

            left = PyIRBinOp(
                line=op.start[0],
                offset=op.start[1],
                op=op.string,
                left=left,
                right=right,
            )

        return left

    @staticmethod
    def _parse_mul(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        left = ExprParser._parse_unary(ts, file_obj)
        while True:
            t = ts.peek()
            if not t or t.type != tokenize.OP or t.string not in {"*", "/", "//", "%"}:
                break

            op = ts.advance()
            right = ExprParser._parse_unary(ts, file_obj)
            if right is None or left is None:
                return left

            left = PyIRBinOp(
                line=op.start[0],
                offset=op.start[1],
                op=op.string,
                left=left,
                right=right,
            )

        return left

    @staticmethod
    def _parse_unary(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        t = ts.peek()
        if t and (
            (t.type == tokenize.OP and t.string in {"+", "-"})
            or (t.type == tokenize.NAME and t.string == "not")
        ):
            op = ts.advance()
            val = ExprParser._parse_unary(ts, file_obj)
            if val is None:
                return None

            op_name = op.string if op.type == tokenize.OP else "not"
            return PyIRUnaryOp(
                line=op.start[0], offset=op.start[1], op=op_name, value=val
            )

        return ExprParser._parse_postfix(ts, file_obj)

    @staticmethod
    def _qual_from_base(base: PyIRNode, file_obj: PyIRFile) -> str | None:
        if isinstance(base, PyIRAtt):
            return base.name

        if isinstance(base, PyIRVarUse):
            n = file_obj.meta.get("id_to_name", {}).get(base.vid)
            if isinstance(n, str):
                return n

        return None

    @staticmethod
    def _is_namespace_head(name: str, file_obj: PyIRFile) -> bool:
        if name in file_obj.meta.get("aliases", {}):
            return True

        if name == "Global":
            return True

        return False

    @staticmethod
    def _parse_postfix(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        base = ExprParser._parse_primary(ts, file_obj)
        if base is None:
            return None

        while True:
            if ts.next_is_op("("):
                lpar = ts.advance()
                callee_name = None
                if isinstance(base, PyIRAtt):
                    callee_name = base.name

                elif isinstance(base, PyIRVarUse):
                    n = file_obj.meta.get("id_to_name", {}).get(base.vid)
                    if isinstance(n, str):
                        callee_name = n

                if isinstance(callee_name, str):
                    callee_name = ExprParser._apply_alias(file_obj, callee_name)
                    args, kwargs = ExprParser._parse_arg_list(ts, file_obj)
                    base = PyIRCall(
                        line=lpar.start[0],
                        offset=lpar.start[1],
                        name=callee_name,
                        args=args,
                        kwargs=kwargs,
                    )
                else:
                    args, kwargs = ExprParser._parse_arg_list(ts, file_obj)
                    base = PyIRCall(
                        line=lpar.start[0],
                        offset=lpar.start[1],
                        name="<callable>",
                        args=[base] + args,
                        kwargs=kwargs,
                    )
                continue

            if ts.next_is_op("."):
                dot = ts.advance()
                attr = ts.peek()
                if not attr or attr.type != tokenize.NAME:
                    break

                ts.advance()
                method = attr.string
                if ts.next_is_op("("):
                    ts.advance()
                    qual = ExprParser._qual_from_base(base, file_obj)
                    if isinstance(base, PyIRVarUse):
                        n = file_obj.meta.get("id_to_name", {}).get(base.vid)
                        if isinstance(n, str) and ExprParser._is_namespace_head(
                            n, file_obj
                        ):
                            callee_name = ExprParser._apply_alias(
                                file_obj, f"{n}.{method}"
                            )
                            args, kwargs = ExprParser._parse_arg_list(ts, file_obj)
                            base = PyIRCall(
                                line=dot.start[0],
                                offset=dot.start[1],
                                name=callee_name,
                                args=args,
                                kwargs=kwargs,
                            )
                            continue

                    if isinstance(base, PyIRAtt):
                        head = base.name
                        callee_name = ExprParser._apply_alias(
                            file_obj, f"{head}.{method}"
                        )
                        args, kwargs = ExprParser._parse_arg_list(ts, file_obj)
                        base = PyIRCall(
                            line=dot.start[0],
                            offset=dot.start[1],
                            name=callee_name,
                            args=args,
                            kwargs=kwargs,
                        )
                        continue

                    args, kwargs = ExprParser._parse_arg_list(ts, file_obj)
                    base = PyIRMethodCall(
                        line=dot.start[0],
                        offset=dot.start[1],
                        obj=base,
                        method=method,
                        args=args,
                        kwargs=kwargs,
                    )

                else:
                    qual = ExprParser._qual_from_base(base, file_obj)
                    if qual is None:
                        break
                    full = f"{qual}.{method}"
                    full = ExprParser._apply_alias(file_obj, full)
                    base = PyIRAtt(line=dot.start[0], offset=dot.start[1], name=full)

                continue

            if ts.next_is_op("["):
                lbr = ts.advance()
                idx = ExprParser._parse_or(ts, file_obj)
                ts.accept_op("]")
                if idx is None:
                    idx = PyIRConstant(
                        line=lbr.start[0],
                        offset=lbr.start[1],
                        value=None,
                    )
                base = PyIRIndex(
                    line=lbr.start[0],
                    offset=lbr.start[1],
                    obj=base,
                    index=idx,
                )
                continue

            break

        return base

    @staticmethod
    def _parse_primary(ts: Tok, file_obj: PyIRFile) -> PyIRNode | None:
        if ts.accept_op("("):
            inner = ExprParser._parse_or(ts, file_obj)
            ts.accept_op(")")
            return inner

        lit = ExprParser._read_literal(ts)
        if lit is not _NO_LIT:
            return PyIRConstant(line=None, offset=None, value=lit)

        name, line, col = ExprParser._read_name(ts)
        if name is not None:
            vid = ExprParser._get_vid(file_obj, name)
            return PyIRVarUse(line=line, offset=col, vid=vid)

        return None

    @staticmethod
    def _parse_arg_list(
        ts: Tok, file_obj: PyIRFile
    ) -> tuple[list[object], dict[str, object]]:
        args: list[object] = []
        kwargs: dict[str, object] = {}
        while not ts.next_is_op(")") and ts.peek() is not None:
            t0 = ts.peek()
            t1 = ts.peek(1)
            if (
                t0
                and t0.type == tokenize.NAME
                and t1
                and t1.type == tokenize.OP
                and t1.string == "="
            ):
                key = ts.advance().string
                ts.advance()
                val = ExprParser._parse_or(ts, file_obj)
                if val is not None:
                    kwargs[key] = val

            else:
                val = ExprParser._parse_or(ts, file_obj)
                if val is not None:
                    args.append(val)

            if ts.next_is_op(","):
                ts.advance()

        ts.accept_op(")")
        return args, kwargs

    @staticmethod
    def _read_literal(ts: Tok):
        t = ts.peek()
        if not t:
            return _NO_LIT

        # bool / None
        if t.type == tokenize.NAME and t.string in ("True", "False", "None"):
            ts.advance()
            return {"True": True, "False": False, "None": None}[t.string]

        # numbers
        if t.type == tokenize.NUMBER:
            ts.advance()
            s = t.string
            try:
                if s.lower().startswith(("0x", "0o", "0b")):
                    return int(s, 0)
                if any(c in s for c in (".", "e", "E")):
                    return float(s)
                return int(s)
            except Exception:
                return s

        # strings
        if t.type == tokenize.STRING:
            ts.advance()
            raw = t.string
            i = 0
            while i < len(raw) and raw[i] not in ("'", '"'):
                i += 1

            if i >= len(raw):
                return raw

            q = raw[i]
            triple = raw[i : i + 3] == q * 3
            if triple:
                return raw[i + 3 : -3] if raw.endswith(q * 3) else raw[i + 3 :]

            return raw[i + 1 : -1] if raw.endswith(q) else raw[i + 1 :]

        return _NO_LIT

    @staticmethod
    def _read_name(ts: Tok) -> tuple[str | None, int, int]:
        t = ts.peek()
        if not t or t.type != tokenize.NAME:
            return (None, 0, 0)

        name = t.string
        line, col = t.start
        ts.advance()
        return (name, line, col)

    @staticmethod
    def _apply_alias(file_obj: PyIRFile, name: str) -> str:
        aliases: dict[str, str] = file_obj.meta.get("aliases", {})
        if not aliases:
            return name

        parts = name.split(".")
        if parts and parts[0] in aliases:
            head = aliases[parts[0]]
            if head:
                parts = head.split(".") + parts[1:]

            else:
                parts = parts[1:]

        return ".".join(parts)

    @staticmethod
    def _get_vid(file_obj: PyIRFile, name: str) -> int:
        n2i: dict[str, int] = file_obj.meta.setdefault("name_to_id", {})
        i2n: dict[int, str] = file_obj.meta.setdefault("id_to_name", {})
        if name in n2i:
            return n2i[name]

        vid = len(n2i)
        n2i[name] = vid
        i2n[vid] = name
        return vid


# endregion


# region parser_statement
class StatementParser:
    @staticmethod
    def parse(tokens: list[tokenize.TokenInfo], file_obj: PyIRFile) -> list[PyIRNode]:
        ts = Tok(tokens)
        out: list[PyIRNode] = []
        if StatementParser._is_assignment(ts):
            target = ts.advance()
            eq = ts.advance()
            rhs = ExprParser.parse(ts, file_obj)
            if rhs is None:
                return out

            name = target.string
            n2i: dict[str, int] = file_obj.meta.setdefault("name_to_id", {})
            created = name not in n2i
            vid = ExprParser._get_vid(file_obj, name)
            is_global = False
            external = False
            is_gcall = False
            g_value = None
            if isinstance(rhs, PyIRCall) and rhs.name == "Global.var":
                is_gcall = True
                args = rhs.args
                kwargs = rhs.kwargs

            elif isinstance(rhs, PyIRMethodCall):
                obj = rhs.obj
                if isinstance(obj, PyIRVarUse):
                    oname = file_obj.meta.get("id_to_name", {}).get(obj.vid)
                    if oname == "Global" and rhs.method == "var":
                        is_gcall = True
                        args = rhs.args
                        kwargs = rhs.kwargs

            if is_gcall:
                is_global = True
                if args:  # type: ignore
                    g_value = args[0]

                elif "value" in kwargs:  # type: ignore
                    g_value = kwargs["value"]  # type: ignore

                else:
                    g_value = PyIRConstant(line=None, offset=None, value=None)

                ext_val = kwargs.get("external", None)  # type: ignore

                if isinstance(ext_val, PyIRConstant) and isinstance(
                    ext_val.value, bool
                ):
                    external = ext_val.value

                elif isinstance(ext_val, bool):
                    external = ext_val

                rhs = g_value  # type: ignore
                gmap = file_obj.meta.setdefault("globals", {})
                gmap[vid] = {"external": external}

            if created:
                out.append(
                    PyIRVarCreate(
                        line=target.start[0],
                        offset=target.start[1],
                        vid=vid,
                        is_global=is_global,
                        external=external,
                    )
                )

            out.append(
                PyIRAssign(
                    line=eq.start[0],
                    offset=eq.start[1],
                    target_vid=vid,
                    value=rhs,  # type: ignore
                )
            )
            return out

        expr = ExprParser.parse(ts, file_obj)
        if expr is not None:
            out.append(expr)

        return out

    @staticmethod
    def _is_assignment(ts: Tok) -> bool:
        t0 = ts.peek()
        t1 = ts.peek(1)
        if not t0 or not t1:
            return False

        if t0.type != tokenize.NAME:
            return False

        return t1.type == tokenize.OP and t1.string == "="


# endregion


# region imports_analyzer
class ImportAnalyzer:
    @staticmethod
    def build(
        file_obj: PyIRFile, node_list: list[PublicLogicNode], start: int
    ) -> tuple[int, list[PyIRNode]]:
        node = node_list[start]
        origin = node.origin
        if origin is None:
            raise ValueError("Origin is None")

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
                    file_obj.meta["aliases"][alias] = real_name
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
                    file_obj.meta["aliases"][alias] = real_name
                    i += 2
                    continue

                modules.append(f"{pkg}.{t}")
                i += 1

        else:
            return (1, [])

        src = Py2GluaConfig.data.get("source", str(file_obj.path.parent))
        project_root = Path(src).resolve()
        stdlib_dir = Path(sysconfig.get_paths()["stdlib"]).resolve()
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
            file_obj.meta["imports"].append(mod)
            ir_nodes.append(ir_node)

        return (1, ir_nodes)  # pyright: ignore[reportReturnType]

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
            mod_path = Path(spec.origin).resolve()  # type: ignore
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
                "name_to_id": {},
                "id_to_name": {},
                "globals": {},
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
            PublicLogicKind.FUNCTION: None,
            PublicLogicKind.CLASS: None,
            PublicLogicKind.BRANCH: None,
            PublicLogicKind.LOOP: None,
            PublicLogicKind.TRY: None,
            PublicLogicKind.WITH: None,
            PublicLogicKind.IMPORT: lambda f, n, i: ImportAnalyzer.build(f, n, i),
            PublicLogicKind.DELETE: None,
            PublicLogicKind.RETURN: None,
            PublicLogicKind.PASS: None,
            PublicLogicKind.COMMENT: None,
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

        ir_nodes = StatementParser.parse(origin.tokens, file_obj)
        return (1, ir_nodes)

    # endregion


# endregion

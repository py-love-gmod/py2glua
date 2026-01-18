from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Final

from ..glua import nil
from .py.ir_dataclass import (
    PyAugAssignType,
    PyBinOPType,
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBackendExpr,
    PyIRBackendKind,
    PyIRBinOP,
    PyIRBreak,
    PyIRCall,
    PyIRComment,
    PyIRConstant,
    PyIRContinue,
    PyIRDict,
    PyIRDictItem,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRList,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
    PyIRSet,
    PyIRSubscript,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarCreate,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
    PyUnaryOPType,
)


@dataclass(slots=True)
class EmitResult:
    code: str


class LuaEmitter:
    _INDENT: Final[str] = "    "
    _LEAK_PREFIX: Final[str] = "<PYTHON_LEAK:"
    _LEAK_SUFFIX: Final[str] = ">"

    _INT_RE: Final[re.Pattern[str]] = re.compile(r"^[+-]?\d[\d_]*\Z")
    _FLOAT_RE: Final[re.Pattern[str]] = re.compile(
        r"^[+-]?\d[\d_]*\.\d[\d_]*([eE][+-]?\d[\d_]*)?\Z|^[+-]?\d[\d_]*([eE][+-]?\d[\d_]*)\Z"
    )

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._indent: int = 0
        self._prev_top_kind: str | None = None

    def emit_file(self, ir: PyIRFile) -> EmitResult:
        self._buf = []
        self._indent = 0
        self._prev_top_kind = None

        for node in ir.body:
            self._stmt(node, top_level=True)

        code = "".join(self._buf)
        if code and not code.endswith("\n"):
            code += "\n"

        return EmitResult(code=code)

    def _wl(self, s: str = "") -> None:
        if s:
            self._buf.append(self._INDENT * self._indent + s + "\n")
        else:
            self._buf.append("\n")

    def _indent_push(self) -> None:
        self._indent += 1

    def _indent_pop(self) -> None:
        if self._indent > 0:
            self._indent -= 1

    def _pos(self, node: PyIRNode | None) -> str:
        if node is None:
            return ""
        if node.line is not None and node.offset is not None:
            return f" LINE|OFFSET: {node.line}|{node.offset}"
        if node.line is not None:
            return f" LINE: {node.line}"
        return ""

    def _leak(self, what: str, node: PyIRNode | None = None) -> str:
        return f"{self._LEAK_PREFIX} {what}{self._pos(node)} {self._LEAK_SUFFIX}"

    def _maybe_blankline_before(self, kind: str, *, top_level: bool) -> None:
        if not top_level:
            return

        if self._prev_top_kind is None:
            self._prev_top_kind = kind
            return

        if kind != self._prev_top_kind:
            self._wl()

        self._prev_top_kind = kind

    def _blankline_after_block(self) -> None:
        if self._indent == 0:
            self._wl()

    def _stmt(self, node: PyIRNode, *, top_level: bool = False) -> None:
        if isinstance(node, PyIRComment):
            self._maybe_blankline_before("comment", top_level=top_level)
            self._emit_comment(node)
            return

        if isinstance(node, PyIRImport):
            self._maybe_blankline_before("import", top_level=top_level)
            self._wl(f"-- {self._leak('import ' + '.'.join(node.modules), node)}")
            return

        if isinstance(node, PyIRPass):
            self._maybe_blankline_before("misc", top_level=top_level)
            self._wl("-- pass")
            return

        if isinstance(node, PyIRVarCreate):
            self._maybe_blankline_before("misc", top_level=top_level)
            if not node.is_global:
                self._wl(f"local {node.name}")

            return

        if isinstance(node, PyIRAssign):
            self._maybe_blankline_before("stmt", top_level=top_level)
            self._wl(f"{self._targets(node.targets)} = {self._expr(node.value)}")
            return

        if isinstance(node, PyIRAugAssign):
            self._maybe_blankline_before("stmt", top_level=top_level)
            self._emit_augassign(node)
            return

        if isinstance(node, PyIRFunctionDef):
            self._maybe_blankline_before("func", top_level=top_level)
            self._emit_function_def(node)
            self._blankline_after_block()
            return

        if isinstance(node, PyIRIf):
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._emit_if(node)
            return

        if isinstance(node, PyIRWhile):
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._emit_while(node)
            return

        if isinstance(node, PyIRFor):
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._emit_for(node)
            return

        if isinstance(node, PyIRReturn):
            self._maybe_blankline_before("stmt", top_level=top_level)
            if self._is_nil(node.value):
                self._wl("return")
            else:
                self._wl(f"return {self._expr(node.value)}")
            return

        if isinstance(node, PyIRBreak):
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._wl("break")
            return

        if isinstance(node, PyIRContinue):
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._wl("continue")
            return

        if isinstance(node, (PyIRWith, PyIRWithItem)):
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._wl(self._leak("with", node))
            return

        if isinstance(node, PyIRCall):
            self._maybe_blankline_before("stmt", top_level=top_level)
            self._wl(self._expr(node))
            return

        self._maybe_blankline_before("misc", top_level=top_level)
        self._wl(self._leak(type(node).__name__, node))

    def _emit_comment(self, node: PyIRComment) -> None:
        text = node.value or ""
        lines = text.splitlines()

        if not lines:
            self._wl("--")
            return

        for line in lines:
            self._wl(f"-- {line.rstrip()}")

    def _emit_function_def(self, fn: PyIRFunctionDef) -> None:
        args = ", ".join(fn.signature.keys()) if fn.signature else ""
        self._wl(f"function {fn.name}({args})")
        self._indent_push()

        if not fn.body:
            self._wl("-- pass")
        else:
            for st in fn.body:
                self._stmt(st, top_level=False)

        self._indent_pop()
        self._wl("end")

    def _emit_if(self, node: PyIRIf) -> None:
        self._wl(f"if {self._expr(node.test)} then")
        self._indent_push()
        for st in node.body or []:
            self._stmt(st, top_level=False)

        self._indent_pop()

        if node.orelse:
            self._wl("else")
            self._indent_push()
            for st in node.orelse:
                self._stmt(st, top_level=False)

            self._indent_pop()

        self._wl("end")

    def _emit_while(self, node: PyIRWhile) -> None:
        self._wl(f"while {self._expr(node.test)} do")
        self._indent_push()
        for st in node.body or []:
            self._stmt(st, top_level=False)

        self._indent_pop()
        self._wl("end")

    def _emit_for(self, node: PyIRFor) -> None:
        rng = self._match_range(node.iter)
        if rng is not None:
            target = self._expr(node.target)
            start_expr, stop_expr = rng
            end_inline = self._try_inline_range_end(stop_expr)
            end_expr = end_inline if end_inline is not None else f"({stop_expr} - 1)"
            self._wl(f"for {target} = {start_expr}, {end_expr} do")
            self._indent_push()
            for st in node.body or []:
                self._stmt(st, top_level=False)

            self._indent_pop()
            self._wl("end")
            return

        self._wl(f"for {self._expr(node.target)} in {self._expr(node.iter)} do")
        self._indent_push()
        for st in node.body or []:
            self._stmt(st, top_level=False)

        self._indent_pop()
        self._wl("end")

    def _emit_augassign(self, node: PyIRAugAssign) -> None:
        target = self._expr(node.target)
        value = self._expr(node.value)

        op = self._augassign_op(node.op)
        if op is None:
            self._wl(self._leak(f"augassign {node.op}", node))
            return

        if op.startswith("LEAK:"):
            self._wl(self._leak(op.removeprefix("LEAK:"), node))
            return

        self._wl(f"{target} = ({target} {op} {value})")

    def _targets(self, targets: list[PyIRNode]) -> str:
        return ", ".join(self._expr(t) for t in targets)

    def _expr(self, node: PyIRNode) -> str:
        if isinstance(node, PyIRConstant):
            return self._const(node.value, node)

        if isinstance(node, PyIRVarUse):
            return node.name

        if isinstance(node, PyIRVarCreate):
            return node.name

        if isinstance(node, PyIRAttribute):
            return f"{self._expr(node.value)}.{node.attr}"

        if isinstance(node, PyIRSubscript):
            return f"{self._expr(node.value)}[{self._expr(node.index)}]"

        if isinstance(node, (PyIRList, PyIRTuple)):
            inside = ", ".join(self._expr(e) for e in node.elements)
            return "{" + inside + "}"

        if isinstance(node, PyIRSet):
            return self._leak("set literal", node)

        if isinstance(node, PyIRDict):
            inside = ", ".join(self._dict_item(i) for i in node.items)
            return "{" + inside + "}"

        if isinstance(node, PyIRBinOP):
            return self._binop_expr(node)

        if isinstance(node, PyIRUnaryOP):
            return self._unary_expr(node)

        if isinstance(node, PyIRCall):
            return self._call(node)

        if isinstance(node, PyIRBackendExpr):
            return self._backend_expr(node)

        return self._leak(type(node).__name__, node)

    def _binop_expr(self, node: PyIRBinOP) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)

        if node.op in (
            PyBinOPType.IN,
            PyBinOPType.NOT_IN,
            PyBinOPType.BIT_OR,
            PyBinOPType.BIT_AND,
            PyBinOPType.BIT_XOR,
            PyBinOPType.BIT_LSHIFT,
            PyBinOPType.BIT_RSHIFT,
            PyBinOPType.FLOORDIV,
        ):
            return self._leak(f"binop {node.op.name}", node)

        if node.op == PyBinOPType.IS:
            return f"({left} == {right})"

        if node.op == PyBinOPType.IS_NOT:
            return f"({left} ~= {right})"

        op = self._binop(node.op)
        if op is None:
            return self._leak(f"binop {node.op}", node)

        return f"({left} {op} {right})"

    def _unary_expr(self, node: PyIRUnaryOP) -> str:
        v = self._expr(node.value)

        if node.op == PyUnaryOPType.BIT_INV:
            return self._leak("unary BIT_INV", node)

        op = self._unaryop(node.op)
        if op is None:
            return self._leak(f"unary {node.op}", node)

        return f"({op}{v})" if not op.endswith(" ") else f"({op}{v})"

    def _call(self, node: PyIRCall) -> str:
        if node.args_kw:
            return self._leak("call kwargs", node)

        args = ", ".join(self._expr(a) for a in node.args_p)
        return f"{self._expr(node.func)}({args})"

    def _backend_expr(self, node: PyIRBackendExpr) -> str:
        if node.kind == PyIRBackendKind.CALL:
            args = ", ".join(self._expr(a) for a in node.args_p)
            return f"{node.name}({args})"

        return node.name

    def _dict_item(self, item: PyIRDictItem) -> str:
        return f"[{self._expr(item.key)}] = {self._expr(item.value)}"

    def _const(self, v: object | nil, node: PyIRNode | None = None) -> str:
        v2 = self._normalize_const(v)

        if v2 is nil or v2 is None:
            return "nil"

        if v2 is True:
            return "true"

        if v2 is False:
            return "false"

        if isinstance(v2, (int, float)):
            return str(v2)

        if isinstance(v2, str):
            return self._quote(v2)

        return self._leak(f"const {type(v2).__name__}", node)

    def _normalize_const(self, v: object | nil) -> object | nil:
        if v is nil or v is None or v is True or v is False:
            return v

        if isinstance(v, (int, float)):
            return v

        if not isinstance(v, str):
            return v

        s = v.strip()

        low = s.lower()
        if low == "true":
            return True

        if low == "false":
            return False

        if low == "none":
            return None

        if low == "nil":
            return None

        if self._INT_RE.match(s):
            try:
                return int(s.replace("_", ""))
            except ValueError:
                return v

        if self._FLOAT_RE.match(s):
            try:
                return float(s.replace("_", ""))
            except ValueError:
                return v

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (bytes, bytearray)):
                return parsed.decode("utf-8", errors="replace")

            return parsed

        except Exception:
            return s

    def _is_nil(self, node: PyIRNode) -> bool:
        if isinstance(node, PyIRConstant):
            v = self._normalize_const(node.value)
            return v is nil or v is None

        return False

    @staticmethod
    def _quote(s: str) -> str:
        s = (
            s.replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
            .replace('"', '\\"')
        )
        return f'"{s}"'

    @staticmethod
    def _binop(op: PyBinOPType) -> str | None:
        mapping: dict[PyBinOPType, str] = {
            PyBinOPType.OR: "or",
            PyBinOPType.AND: "and",
            PyBinOPType.EQ: "==",
            PyBinOPType.NE: "~=",
            PyBinOPType.LT: "<",
            PyBinOPType.GT: ">",
            PyBinOPType.LE: "<=",
            PyBinOPType.GE: ">=",
            PyBinOPType.ADD: "+",
            PyBinOPType.SUB: "-",
            PyBinOPType.MUL: "*",
            PyBinOPType.DIV: "/",
            PyBinOPType.MOD: "%",
            PyBinOPType.POW: "^",
        }
        return mapping.get(op)

    @staticmethod
    def _unaryop(op: PyUnaryOPType) -> str | None:
        mapping: dict[PyUnaryOPType, str] = {
            PyUnaryOPType.PLUS: "+",
            PyUnaryOPType.MINUS: "-",
            PyUnaryOPType.NOT: "not ",
        }
        return mapping.get(op)

    @staticmethod
    def _augassign_op(op: PyAugAssignType) -> str | None:
        mapping: dict[PyAugAssignType, str] = {
            PyAugAssignType.ADD: "+",
            PyAugAssignType.SUB: "-",
            PyAugAssignType.MUL: "*",
            PyAugAssignType.DIV: "/",
            PyAugAssignType.MOD: "%",
            PyAugAssignType.POW: "^",
            PyAugAssignType.FLOORDIV: "LEAK: augassign FLOORDIV",
            PyAugAssignType.BIT_OR: "LEAK: augassign BIT_OR",
            PyAugAssignType.BIT_AND: "LEAK: augassign BIT_AND",
            PyAugAssignType.BIT_XOR: "LEAK: augassign BIT_XOR",
            PyAugAssignType.LSHIFT: "LEAK: augassign LSHIFT",
            PyAugAssignType.RSHIFT: "LEAK: augassign RSHIFT",
        }
        return mapping.get(op)

    def _try_inline_range_end(self, stop_expr: str) -> str | None:
        s = stop_expr.strip()

        if self._INT_RE.match(s):
            try:
                n = int(s.replace("_", ""))
                return str(n - 1)

            except ValueError:
                return None

        if self._FLOAT_RE.match(s):
            try:
                n = float(s.replace("_", ""))
                return None

            except ValueError:
                return None

        return None

    def _match_range(self, iter_expr: PyIRNode) -> tuple[str, str] | None:
        if not isinstance(iter_expr, PyIRCall):
            return None

        if iter_expr.args_kw:
            return None

        if not isinstance(iter_expr.func, PyIRVarUse):
            return None

        if iter_expr.func.name != "range":
            return None

        args = iter_expr.args_p
        if len(args) == 1:
            return ("0", self._expr(args[0]))

        if len(args) == 2:
            return (self._expr(args[0]), self._expr(args[1]))

        return None

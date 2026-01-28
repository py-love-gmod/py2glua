from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .._lang.compiler.passes.analysis.symlinks import PyIRSymLink
from .._lang.compiler.passes.lowering.rewrite_anonymous_functions import PyIRFnExpr
from .py.ir_dataclass import (
    PyAugAssignType,
    PyBinOPType,
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBinOP,
    PyIRBreak,
    PyIRCall,
    PyIRClassDef,
    PyIRComment,
    PyIRConstant,
    PyIRContinue,
    PyIRDecorator,
    PyIRDict,
    PyIREmitExpr,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRList,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
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

    _BIN_OP: Final[dict[PyBinOPType, str]] = {
        PyBinOPType.OR: "or",
        PyBinOPType.AND: "and",
        PyBinOPType.EQ: "==",
        PyBinOPType.NE: "~=",
        PyBinOPType.LT: "<",
        PyBinOPType.GT: ">",
        PyBinOPType.LE: "<=",
        PyBinOPType.GE: ">=",
        PyBinOPType.BIT_OR: "|",
        PyBinOPType.BIT_XOR: "~",
        PyBinOPType.BIT_AND: "&",
        PyBinOPType.BIT_LSHIFT: "<<",
        PyBinOPType.BIT_RSHIFT: ">>",
        PyBinOPType.ADD: "+",
        PyBinOPType.SUB: "-",
        PyBinOPType.MUL: "*",
        PyBinOPType.DIV: "/",
        PyBinOPType.FLOORDIV: "//",
        PyBinOPType.MOD: "%",
        PyBinOPType.POW: "^",
    }

    _UNARY_OP: Final[dict[PyUnaryOPType, str]] = {
        PyUnaryOPType.PLUS: "+",
        PyUnaryOPType.MINUS: "-",
        PyUnaryOPType.NOT: "not",
        PyUnaryOPType.BIT_INV: "~",
    }

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._indent: int = 0
        self._prev_top_kind: str | None = None

    def emit_file(self, ir: PyIRFile) -> EmitResult:
        self._buf.clear()
        self._indent = 0
        self._prev_top_kind = None

        for node in ir.body:
            self._stmt(node, top_level=True)

        code = "".join(self._buf)
        if code and not code.endswith("\n"):
            code += "\n"

        return EmitResult(code=code)

    def _wl(self, s: str = "") -> None:
        # IMPORTANT: support multiline strings; indent every line
        if not s:
            self._buf.append("\n")
            return

        prefix = self._INDENT * self._indent
        lines = s.splitlines()
        for ln in lines:
            self._buf.append(prefix + ln + "\n")

    def _indent_push(self) -> None:
        self._indent += 1

    def _indent_pop(self) -> None:
        self._indent = max(0, self._indent - 1)

    def _leak(self, what: str) -> str:
        return f"-- {self._LEAK_PREFIX} {what} {self._LEAK_SUFFIX}"

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

    def _stmt(self, node: PyIRNode, *, top_level: bool) -> None:
        # after the attach step, decorator statements should not appear;
        # if they still do, ignore them to avoid garbage output
        if isinstance(node, PyIRDecorator):
            return

        if isinstance(node, PyIRComment):
            self._maybe_blankline_before("comment", top_level=top_level)
            self._emit_comment(node)
            return

        if isinstance(node, PyIRImport):
            self._maybe_blankline_before("import", top_level=top_level)
            self._wl(str(node))
            return

        if isinstance(node, PyIRPass):
            self._maybe_blankline_before("misc", top_level=top_level)
            self._wl("-- pass")
            return

        if isinstance(node, PyIRVarCreate):
            self._maybe_blankline_before("decl", top_level=top_level)
            self._wl(str(node))
            return

        if isinstance(node, PyIRAssign):
            self._maybe_blankline_before("stmt", top_level=top_level)
            targets = ", ".join(self._expr(t) for t in node.targets)
            self._wl(f"{targets} = {self._expr(node.value)}")
            return

        if isinstance(node, PyIRAugAssign):
            self._maybe_blankline_before("stmt", top_level=top_level)
            self._emit_augassign(node)
            return

        if isinstance(node, PyIRFunctionDef):
            self._maybe_blankline_before("func", top_level=top_level)
            self._emit_function(node)
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
            self._wl(
                "return" if node.value is None else f"return {self._expr(node.value)}"
            )
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
            self._wl(self._leak("with-statement"))
            return

        if isinstance(
            node,
            (
                PyIRCall,
                PyIRAttribute,
                PyIREmitExpr,
                PyIRConstant,
                PyIRBinOP,
                PyIRUnaryOP,
                PyIRSymLink,
                PyIRFnExpr,
            ),
        ):
            self._maybe_blankline_before("stmt", top_level=top_level)
            self._wl(self._expr(node))
            return

        if isinstance(node, PyIRClassDef):
            self._maybe_blankline_before("class", top_level=top_level)
            self._emit_class(node)
            self._blankline_after_block()
            return

        self._maybe_blankline_before("misc", top_level=top_level)
        self._wl(self._leak(type(node).__name__))

    def _emit_class(self, node: PyIRClassDef) -> None:
        if node.bases:
            bases = ", ".join(self._expr(b) for b in node.bases)
            self._wl(f"-- class {node.name}({bases})")
        else:
            self._wl(f"-- class {node.name}")

        self._wl(f"{node.name} = {{}}")
        self._wl(f"{node.name}.__index = {node.name}")

        for item in node.body:
            if isinstance(item, PyIRFunctionDef):
                self._emit_class_method(node.name, item)
            else:
                self._wl(self._leak(type(item).__name__))

    def _emit_class_method(self, class_name: str, fn: PyIRFunctionDef) -> None:
        args = ", ".join(fn.signature.keys())
        self._wl(f"function {class_name}.{fn.name}({args})")
        self._indent_push()

        if not fn.body:
            self._wl("-- pass")
        else:
            for st in fn.body:
                self._stmt(st, top_level=False)

        self._indent_pop()
        self._wl("end")

    def _emit_comment(self, node: PyIRComment) -> None:
        for line in (node.value or "").splitlines() or [""]:
            self._wl(f"-- {line}")

    def _emit_function(self, fn: PyIRFunctionDef) -> None:
        args = ", ".join(fn.signature.keys())
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
        for st in node.body:
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
        for st in node.body:
            self._stmt(st, top_level=False)
        self._indent_pop()
        self._wl("end")

    def _emit_for(self, node: PyIRFor) -> None:
        self._wl(f"for {self._expr(node.target)} in {self._expr(node.iter)} do")
        self._indent_push()
        for st in node.body:
            self._stmt(st, top_level=False)
        self._indent_pop()
        self._wl("end")

    def _emit_augassign(self, node: PyIRAugAssign) -> None:
        op = self._augassign_op(node.op)
        if op is None:
            self._wl(self._leak(f"augassign {node.op.name}"))
            return

        t = self._expr(node.target)
        v = self._expr(node.value)
        self._wl(f"{t} = ({t} {op} {v})")

    def _emit_fn_expr(self, node: PyIRFnExpr) -> str:
        args = ", ".join(node.signature.keys())
        lines: list[str] = [f"function({args})"]

        old_buf = self._buf
        old_indent = self._indent
        old_prev = self._prev_top_kind

        tmp: list[str] = []
        self._buf = tmp
        self._prev_top_kind = None
        self._indent = 1

        if not node.body:
            self._wl("-- pass")

        else:
            for st in node.body:
                self._stmt(st, top_level=False)

        self._buf = old_buf
        self._indent = old_indent
        self._prev_top_kind = old_prev

        body = "".join(tmp).rstrip("\n")
        if body:
            lines.extend(body.splitlines())

        lines.append("end")
        return "\n".join(lines)

    def _expr(self, node: PyIRNode) -> str:
        if isinstance(node, PyIRFnExpr):
            return self._emit_fn_expr(node)

        if isinstance(node, PyIRAttribute):
            return f"{self._expr(node.value)}.{node.attr}"

        if isinstance(
            node,
            (
                PyIRConstant,
                PyIRVarUse,
                PyIRVarCreate,
                PyIRSubscript,
                PyIREmitExpr,
                PyIRSymLink,
            ),
        ):
            return str(node)

        if isinstance(node, PyIRCall):
            return self._call(node)

        if isinstance(node, PyIRBinOP):
            op = self._BIN_OP.get(node.op)
            if op is None:
                return self._leak(f"binop {node.op.name}")
            return f"({self._expr(node.left)} {op} {self._expr(node.right)})"

        if isinstance(node, PyIRUnaryOP):
            op = self._UNARY_OP.get(node.op)
            if op is None:
                return self._leak(f"unary {node.op.name}")
            v = self._expr(node.value)
            return f"(not {v})" if op == "not" else f"({op}{v})"

        if isinstance(node, PyIRList):
            return self._emit_list(node)

        if isinstance(node, PyIRDict):
            return self._emit_dict(node)

        if isinstance(node, PyIRTuple):
            return self._emit_tuple(node)

        return self._leak(type(node).__name__)

    def _call(self, node: PyIRCall) -> str:
        if node.args_kw:
            return self._leak("call with kwargs")

        args = ", ".join(self._expr(a) for a in node.args_p)

        func_s = self._expr(node.func)
        if isinstance(node.func, PyIRFnExpr):
            func_s = f"({func_s})"

        return f"{func_s}({args})"

    def _emit_list(self, node: PyIRList) -> str:
        if not node.elements:
            return "{}"
        items = ", ".join(self._expr(e) for e in node.elements)
        return f"{{ {items} }}"

    def _emit_dict(self, node: PyIRDict) -> str:
        if not node.items:
            return "{}"

        parts: list[str] = []
        for item in node.items:
            key = item.key
            val = item.value

            if isinstance(key, PyIRConstant) and isinstance(key.value, str):
                k = f"[{repr(key.value)}]"
            elif isinstance(key, PyIRConstant) and isinstance(key.value, (int, float)):
                k = f"[{key.value}]"
            else:
                k = f"[{self._expr(key)}]"

            parts.append(f"{k} = {self._expr(val)}")

        return "{ " + ", ".join(parts) + " }"

    def _emit_tuple(self, node: PyIRTuple) -> str:
        if not node.elements:
            return ""
        return ", ".join(self._expr(e) for e in node.elements)

    @staticmethod
    def _augassign_op(op: PyAugAssignType) -> str | None:
        return {
            PyAugAssignType.ADD: "+",
            PyAugAssignType.SUB: "-",
            PyAugAssignType.MUL: "*",
            PyAugAssignType.DIV: "/",
            PyAugAssignType.MOD: "%",
            PyAugAssignType.POW: "^",
        }.get(op)

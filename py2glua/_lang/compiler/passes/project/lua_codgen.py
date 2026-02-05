from __future__ import annotations

import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

from .....config import Py2GluaConfig
from ....compiler.compiler_ir import PyIRDo, PyIRFunctionExpr, PyIRGoto, PyIRLabel
from ....py.ir_dataclass import (
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
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


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

        self._pending_locals: list[PyIRVarCreate] = []

        self._locals_stack: list[set[str]] = [set()]

    def emit_file(self, ir: PyIRFile) -> EmitResult:
        self._buf.clear()
        self._indent = 0
        self._prev_top_kind = None
        self._pending_locals.clear()
        self._locals_stack = [set()]

        for node in ir.body:
            self._stmt(node, top_level=True)

        self._flush_pending_locals(top_level=True)

        code = "".join(self._buf)
        if code and not code.endswith("\n"):
            code += "\n"

        return EmitResult(code=code)

    def _wl(self, s: str = "") -> None:
        if not s:
            self._buf.append("\n")
            return

        prefix = self._INDENT * self._indent
        for ln in s.splitlines():
            self._buf.append(prefix + ln + "\n")

    def _indent_push(self) -> None:
        self._indent += 1

    def _indent_pop(self) -> None:
        self._indent = max(0, self._indent - 1)

    def _scope_push(self) -> None:
        self._locals_stack.append(set())

    def _scope_pop(self) -> None:
        if len(self._locals_stack) > 1:
            self._locals_stack.pop()

    def _scope_add_local(self, name: str) -> None:
        self._locals_stack[-1].add(name)

    def _scope_has_local(self, name: str) -> bool:
        return any(name in s for s in reversed(self._locals_stack))

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

    @staticmethod
    def _varcreate_is_local(node: PyIRVarCreate) -> bool:
        """
        Support both old and new PyIRVarCreate field sets:
          - old: is_global (False => local)
          - new: is_local (True => local)
        Fallback: treat as local.
        """
        fset = {f.name for f in fields(node)}
        if "is_global" in fset:
            return not bool(getattr(node, "is_global"))
        if "is_local" in fset:
            return bool(getattr(node, "is_local"))
        return True

    @staticmethod
    def _emit_varcreate_line(node: PyIRVarCreate) -> str:
        if LuaEmitter._varcreate_is_local(node):
            return f"local {node.name}"
        return node.name

    def _pop_pending_local(self, name: str) -> bool:
        for i, n in enumerate(self._pending_locals):
            if n.name == name:
                self._pending_locals.pop(i)
                return True
        return False

    def _flush_pending_locals(self, *, top_level: bool) -> None:
        if not self._pending_locals:
            return

        self._maybe_blankline_before("decl", top_level=top_level)
        for n in self._pending_locals:
            self._wl(self._emit_varcreate_line(n))
        self._pending_locals.clear()

    def _try_merge_local_assign(self, node: PyIRAssign, *, top_level: bool) -> bool:
        """
        local x
        x = expr
        =>
        local x = expr
        """
        if len(node.targets) != 1:
            return False

        t = node.targets[0]
        if not isinstance(t, PyIRVarUse):
            return False

        name = t.name
        if not self._pop_pending_local(name):
            return False

        self._flush_pending_locals(top_level=top_level)

        self._maybe_blankline_before("stmt", top_level=top_level)
        self._wl(f"local {name} = {self._expr(node.value)}")
        return True

    def _stmt(self, node: PyIRNode, *, top_level: bool) -> None:
        if isinstance(node, PyIRDecorator):
            return

        if isinstance(node, PyIRVarCreate):
            if self._varcreate_is_local(node):
                self._pending_locals.append(node)
                self._scope_add_local(node.name)
            else:
                self._flush_pending_locals(top_level=top_level)
                self._maybe_blankline_before("decl", top_level=top_level)
                self._wl(self._emit_varcreate_line(node))
            return

        if isinstance(node, PyIRAssign):
            if self._try_merge_local_assign(node, top_level=top_level):
                return
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("stmt", top_level=top_level)
            targets = ", ".join(self._expr(t) for t in node.targets)
            self._wl(f"{targets} = {self._expr(node.value)}")
            return

        if isinstance(node, PyIRComment):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("comment", top_level=top_level)
            self._emit_comment(node)
            return

        if isinstance(node, PyIRImport):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("import", top_level=top_level)
            self._wl(str(node))
            return

        if isinstance(node, PyIRPass):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("misc", top_level=top_level)
            self._wl("-- pass")
            return

        if isinstance(node, PyIRAugAssign):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("stmt", top_level=top_level)
            self._emit_augassign(node)
            return

        if isinstance(node, PyIRFunctionDef):
            local_init = self._pop_pending_local(node.name)
            had_local = local_init or self._scope_has_local(node.name)

            self._flush_pending_locals(top_level=top_level)

            self._maybe_blankline_before("func", top_level=top_level)
            self._emit_function(
                node,
                local_init=local_init,
                assign_style=(had_local and not local_init),
            )
            self._blankline_after_block()
            return

        if isinstance(node, PyIRClassDef):
            local_init = self._pop_pending_local(node.name)
            self._flush_pending_locals(top_level=top_level)

            self._maybe_blankline_before("class", top_level=top_level)
            self._emit_class(node, local_init=local_init)
            self._blankline_after_block()
            return

        if isinstance(node, PyIRDo):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._emit_do(node)
            self._blankline_after_block()
            return

        if isinstance(node, PyIRLabel):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._wl(f"::{node.name}::")
            return

        if isinstance(node, PyIRGoto):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._wl(f"goto {node.label}")
            return

        if isinstance(node, PyIRIf):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._emit_if(node)
            return

        if isinstance(node, PyIRWhile):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._emit_while(node)
            return

        if isinstance(node, PyIRFor):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._emit_for(node)
            return

        if isinstance(node, PyIRReturn):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("stmt", top_level=top_level)
            if node.value is None:
                self._wl("return")
            else:
                self._wl(f"return {self._expr(node.value)}")
            return

        if isinstance(node, PyIRBreak):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._wl("break")
            return

        if isinstance(node, PyIRContinue):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("ctrl", top_level=top_level)
            self._wl("continue")
            return

        if isinstance(node, (PyIRWith, PyIRWithItem)):
            self._flush_pending_locals(top_level=top_level)
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
                PyIRFunctionExpr,
            ),
        ):
            self._flush_pending_locals(top_level=top_level)
            self._maybe_blankline_before("stmt", top_level=top_level)
            self._wl(self._expr(node))
            return

        self._flush_pending_locals(top_level=top_level)
        self._maybe_blankline_before("misc", top_level=top_level)
        self._wl(self._leak(type(node).__name__))

    def _emit_do(self, node: PyIRDo) -> None:
        self._wl("do")
        self._indent_push()
        self._scope_push()
        for st in node.body:
            self._stmt(st, top_level=False)
        self._flush_pending_locals(top_level=False)
        self._scope_pop()
        self._indent_pop()
        self._wl("end")

    def _emit_class(self, node: PyIRClassDef, *, local_init: bool) -> None:
        if node.bases:
            bases = ", ".join(self._expr(b) for b in node.bases)
            self._wl(f"-- class {node.name}({bases})")
        else:
            self._wl(f"-- class {node.name}")

        if local_init:
            self._wl(f"local {node.name} = {{}}")
            self._scope_add_local(node.name)
        else:
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
        self._scope_push()

        if not fn.body:
            self._wl("-- pass")
        else:
            for st in fn.body:
                self._stmt(st, top_level=False)

        self._flush_pending_locals(top_level=False)
        self._scope_pop()
        self._indent_pop()
        self._wl("end")

    def _emit_comment(self, node: PyIRComment) -> None:
        for line in (node.value or "").splitlines() or [""]:
            self._wl(f"-- {line}")

    def _emit_function(
        self,
        fn: PyIRFunctionDef,
        *,
        local_init: bool,
        assign_style: bool,
    ) -> None:
        args = ", ".join(fn.signature.keys())

        if local_init:
            self._wl(f"local function {fn.name}({args})")
            self._scope_add_local(fn.name)

        elif assign_style:
            self._wl(f"{fn.name} = function({args})")

        else:
            self._wl(f"function {fn.name}({args})")

        self._indent_push()
        self._scope_push()

        if not fn.body:
            self._wl("-- pass")
        else:
            for st in fn.body:
                self._stmt(st, top_level=False)

        self._flush_pending_locals(top_level=False)

        self._scope_pop()
        self._indent_pop()
        self._wl("end")

    def _emit_if(self, node: PyIRIf) -> None:
        self._wl(f"if {self._expr(node.test)} then")
        self._indent_push()
        self._scope_push()
        for st in node.body:
            self._stmt(st, top_level=False)
        self._flush_pending_locals(top_level=False)
        self._scope_pop()
        self._indent_pop()

        if node.orelse:
            self._wl("else")
            self._indent_push()
            self._scope_push()
            for st in node.orelse:
                self._stmt(st, top_level=False)
            self._flush_pending_locals(top_level=False)
            self._scope_pop()
            self._indent_pop()

        self._wl("end")

    def _emit_while(self, node: PyIRWhile) -> None:
        self._wl(f"while {self._expr(node.test)} do")
        self._indent_push()
        self._scope_push()
        for st in node.body:
            self._stmt(st, top_level=False)
        self._flush_pending_locals(top_level=False)
        self._scope_pop()
        self._indent_pop()
        self._wl("end")

    def _emit_for(self, node: PyIRFor) -> None:
        self._wl(f"for {self._expr(node.target)} in {self._expr(node.iter)} do")
        self._indent_push()
        self._scope_push()
        for st in node.body:
            self._stmt(st, top_level=False)
        self._flush_pending_locals(top_level=False)
        self._scope_pop()
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

    def _expr(self, node: PyIRNode) -> str:
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

        if isinstance(node, PyIRFunctionExpr):
            return self._emit_function_expr(node)

        return self._leak(type(node).__name__)

    def _emit_function_expr(self, fn: PyIRFunctionExpr) -> str:
        args = ", ".join(fn.signature.keys())
        if not fn.body:
            return f"function({args}) end"

        sub = LuaEmitter()
        sub._indent = 1
        sub._scope_push()
        for st in fn.body:
            sub._stmt(st, top_level=False)
        sub._flush_pending_locals(top_level=False)
        sub._scope_pop()

        body = "".join(sub._buf).rstrip("\n")
        return f"function({args})\n{body}\nend"

    def _call(self, node: PyIRCall) -> str:
        if node.args_kw:
            return self._leak("call with kwargs")

        args = ", ".join(self._expr(a) for a in node.args_p)

        if isinstance(node.func, PyIRFunctionExpr):
            return f"({self._expr(node.func)})({args})"

        func_s = self._expr(node.func)
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


class EmitLuaProjectPass:
    """
    Project-pass: emits Lua and writes files into Py2GluaConfig.output.

    IMPORTANT:
      - Before writing, it fully deletes Py2GluaConfig.output directory (clean build).
      - Then recreates it and writes into: output/lua/<rel>.lua

    Ghost PyIRFiles supported:
      - if ir.path is not under source, but starts with 'lua/', it is treated as virtual lua path.
      - else fallback by module name via ctx.module_name_by_path.
    """

    _OUT_LUA_DIR: Final[str] = "lua"

    @classmethod
    def run(cls, files: list[PyIRFile], ctx: SymLinkContext) -> list[PyIRFile]:
        out_dir = Py2GluaConfig.output.resolve()
        cls._safe_clean_output_dir(out_dir)

        out_root = out_dir / cls._OUT_LUA_DIR
        out_root.mkdir(parents=True, exist_ok=True)

        ctx.ensure_module_index()
        cls._add_resolved_path_aliases(ctx)

        emitter = LuaEmitter()
        used_out: dict[Path, Path] = {}

        for ir in files:
            if ir.path is None:
                raise RuntimeError("EmitLuaProjectPass: IR file has no path.")

            rel = cls._lua_rel_path(ir.path, ctx)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            norm = out_path.resolve()
            prev = used_out.get(norm)
            if prev is not None:
                raise RuntimeError(
                    "EmitLuaProjectPass: output path collision.\n"
                    f"Output: {norm}\n"
                    f"A: {prev}\n"
                    f"B: {ir.path}"
                )
            used_out[norm] = ir.path

            code = emitter.emit_file(ir).code
            out_path.write_text(code, encoding="utf-8")

        return files

    @classmethod
    def _safe_clean_output_dir(cls, out_dir: Path) -> None:
        """
        Fully deletes output directory before build.
        Includes basic safety checks to avoid nuking filesystem roots.
        """
        if out_dir.exists() and not out_dir.is_dir():
            raise RuntimeError(f"Output path exists but is not a directory: {out_dir}")

        resolved = out_dir.resolve()
        anchor = Path(resolved.anchor) if resolved.anchor else None
        if anchor is not None and resolved == anchor:
            raise RuntimeError(
                f"Refusing to delete output directory at filesystem root: {resolved}"
            )

        if len(resolved.parts) <= 1:
            raise RuntimeError(f"Refusing to delete output directory: {resolved}")

        if resolved.exists():
            shutil.rmtree(resolved)

        resolved.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _lua_rel_path(cls, p: Path, ctx: SymLinkContext) -> Path:
        try:
            src_root = Py2GluaConfig.source.resolve()
            rel = p.resolve().relative_to(src_root)

            if rel.parts and rel.parts[0] == "lua":
                rel = rel.relative_to("lua")

            return rel.with_suffix(".lua")
        except Exception:
            pass

        try:
            if p.parts and p.parts[0] == "lua":
                rel2 = p.relative_to("lua")
                return rel2.with_suffix(".lua")

        except Exception:
            pass

        mod = cls._module_of_path(p, ctx)
        if not mod:
            raise RuntimeError(
                f"EmitLuaProjectPass: cannot compute output path for: {p!s}"
            )

        return Path(*mod.split(".")).with_suffix(".lua")

    @classmethod
    def _add_resolved_path_aliases(cls, ctx: SymLinkContext) -> None:
        add: dict[Path, str] = {}
        for path, mod in list(ctx.module_name_by_path.items()):
            try:
                pr = path.resolve()
            except Exception:
                continue
            if pr not in ctx.module_name_by_path:
                add[pr] = mod
        if add:
            ctx.module_name_by_path.update(add)

    @staticmethod
    def _module_of_path(p: Path, ctx: SymLinkContext) -> str | None:
        hit = ctx.module_name_by_path.get(p)
        if hit is not None:
            return hit

        try:
            pr = p.resolve()

        except Exception:
            return None

        return ctx.module_name_by_path.get(pr)

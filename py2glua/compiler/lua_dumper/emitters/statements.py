"""Эмиттеры инструкций, регистрируются в глобальном _EMITTERS."""

from __future__ import annotations

from plg_reader import (
    IRAnnotatedAssign,
    IRAssign,
    IRBreak,
    IRComment,
    IRContinue,
    IRDelete,
    IRExprStatement,
    IRFor,
    IRIf,
    IRPass,
    IRRaise,
    IRReturn,
    IRTry,
    IRWhile,
    IRWith,
)

from .._core import _EMITTERS, _INDENT_STR
from .._core import emit as _emit
from ..lua_helpers import expr_to_str


def register():
    def emit_assign(node: IRAssign, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        targets_str = ", ".join(expr_to_str(t) for t in node.targets)
        value_str = expr_to_str(node.value)
        op = ""
        if node.is_aug and node.aug_op:
            from ..lua_helpers import binop_to_lua

            op = binop_to_lua(node.aug_op) + " "

        lines.append(f"{prefix}{targets_str} = {op}{value_str}")

    def emit_annotated_assign(node: IRAnnotatedAssign, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        ann_str = expr_to_str(node.annotation) if node.annotation else "?"
        target_str = expr_to_str(node.target)
        value_str = expr_to_str(node.value) if node.value else "nil"
        lines.append(f"{prefix}{target_str} = {value_str}  -- : {ann_str}")

    def emit_if(node: IRIf, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        test_str = expr_to_str(node.test)
        lines.append(f"{prefix}if {test_str} then")
        for stmt in node.body:
            _emit(stmt, lines, indent + 1)

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], IRIf):
                elif_node = node.orelse[0]
                test_str2 = expr_to_str(elif_node.test)
                lines.append(f"{prefix}elseif {test_str2} then")
                for stmt in elif_node.body:
                    _emit(stmt, lines, indent + 1)

            else:
                lines.append(f"{prefix}else")
                for stmt in node.orelse:
                    _emit(stmt, lines, indent + 1)

        lines.append(f"{prefix}end")

    def emit_while(node: IRWhile, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        test_str = expr_to_str(node.test)
        lines.append(f"{prefix}while {test_str} do")
        for stmt in node.body:
            _emit(stmt, lines, indent + 1)

        lines.append(f"{prefix}end")

    def emit_for(node: IRFor, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        target_str = expr_to_str(node.target)
        iter_str = expr_to_str(node.iter)
        lines.append(f"{prefix}for {target_str} in {iter_str} do")
        for stmt in node.body:
            _emit(stmt, lines, indent + 1)

        lines.append(f"{prefix}end")

    def emit_return(node: IRReturn, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        val = expr_to_str(node.value) if node.value else ""
        lines.append(f"{prefix}return {val}".rstrip())

    def emit_break(node: IRBreak, lines: list[str], indent: int):
        lines.append(f"{_INDENT_STR * indent}break")

    def emit_continue(node: IRContinue, lines: list[str], indent: int):
        lines.append(f"{_INDENT_STR * indent}-- continue (unsupported)")

    def emit_pass(node: IRPass, lines: list[str], indent: int):
        lines.append(f"{_INDENT_STR * indent}-- pass")

    def emit_expr_stmt(node: IRExprStatement, lines: list[str], indent: int):
        lines.append(f"{_INDENT_STR * indent}{expr_to_str(node.expr)}")

    def emit_delete(node: IRDelete, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        targets_str = ", ".join(expr_to_str(t) for t in node.targets)
        lines.append(f"{prefix}-- del {targets_str} (set to nil)")
        lines.append(f"{prefix}{targets_str} = nil")

    def emit_raise(node: IRRaise, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        exc_str = expr_to_str(node.exc) if node.exc else ""
        cause_str = f" from {expr_to_str(node.cause)}" if node.cause else ""
        lines.append(f"{prefix}error({exc_str}{cause_str})")

    def emit_try(node: IRTry, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        lines.append(f"{prefix}-- try (using pcall/xpcall)")
        lines.append(f"{prefix}do")
        for stmt in node.body:
            _emit(stmt, lines, indent + 1)

        lines.append(f"{prefix}end -- pcall would go here")
        if node.handlers:
            lines.append(f"{prefix}-- except handlers:")
            for h in node.handlers:
                exc_type = expr_to_str(h.type) if h.type else "Exception"
                name = h.name if h.name else "_"
                lines.append(f"{prefix}-- catch {exc_type} as {name}")

        if node.orelse:
            lines.append(f"{prefix}-- else:")
            for stmt in node.orelse:
                _emit(stmt, lines, indent + 1)

        if node.finalbody:
            lines.append(f"{prefix}-- finally:")
            for stmt in node.finalbody:
                _emit(stmt, lines, indent + 1)

    def emit_with(node: IRWith, lines: list[str], indent: int):
        prefix = _INDENT_STR * indent
        for item in node.items:
            ctx_str = expr_to_str(item.context_expr)
            if item.optional_vars:
                var_str = expr_to_str(item.optional_vars)
                lines.append(f"{prefix}-- with {ctx_str} as {var_str}:")

            else:
                lines.append(f"{prefix}-- with {ctx_str}:")

        for stmt in node.body:
            _emit(stmt, lines, indent)

    def emit_comment(node: IRComment, lines: list[str], indent: int):
        lines.append(f"{_INDENT_STR * indent}-- {node.text}")

    _EMITTERS[IRAssign] = emit_assign
    _EMITTERS[IRAnnotatedAssign] = emit_annotated_assign
    _EMITTERS[IRIf] = emit_if
    _EMITTERS[IRWhile] = emit_while
    _EMITTERS[IRFor] = emit_for
    _EMITTERS[IRReturn] = emit_return
    _EMITTERS[IRBreak] = emit_break
    _EMITTERS[IRContinue] = emit_continue
    _EMITTERS[IRPass] = emit_pass
    _EMITTERS[IRExprStatement] = emit_expr_stmt
    _EMITTERS[IRDelete] = emit_delete
    _EMITTERS[IRRaise] = emit_raise
    _EMITTERS[IRTry] = emit_try
    _EMITTERS[IRWith] = emit_with
    _EMITTERS[IRComment] = emit_comment

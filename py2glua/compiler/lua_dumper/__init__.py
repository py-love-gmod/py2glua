# FIXME
from __future__ import annotations

from typing import Callable

from plg_reader import (
    IRAnnotatedAssign,
    IRAssign,
    IRAttribute,
    IRBinOp,
    IRBinOpType,
    IRBreak,
    IRCall,
    IRClassDef,
    IRComment,
    IRConstant,
    IRContinue,
    IRDelete,
    IRDict,
    IRExprStatement,
    IRFile,
    IRFor,
    IRFString,
    IRFunctionDef,
    IRIf,
    IRIfExpr,
    IRList,
    IRName,
    IRNode,
    IRPass,
    IRRaise,
    IRReturn,
    IRSet,
    IRSubscript,
    IRTry,
    IRTuple,
    IRUnaryOp,
    IRWhile,
    IRWith,
)

from ..analysis import IRSymbolRef, SymbolTable

_INDENT = "    "


_EMITTERS: dict[type, Callable] = {}


def register(node_type: type):
    """Декоратор для регистрации функции-эмиттера."""

    def decorator(func):
        _EMITTERS[node_type] = func
        return func

    return decorator


def emit_node(node: IRNode, lines: list[str], indent: int) -> None:
    """Динамически выбирает и вызывает подходящий эмиттер."""
    emitter = _EMITTERS.get(type(node))
    if emitter is not None:
        emitter(node, lines, indent)

    else:
        prefix = _INDENT * indent
        lines.append(f"{prefix}-- [[ {type(node).__name__} ]] {node}")


@register(IRFunctionDef)
def emit_function_def(node: IRFunctionDef, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent

    for dec in node.decorators:
        lines.append(f"{prefix}-- @{expr_to_str(dec.expr)}")

    params = []
    for p in node.params:
        if p.kind == "star_arg":
            params.append(f"*{p.name}")
        elif p.kind == "kw_arg":
            params.append(f"**{p.name}")
        elif p.kind in ("slash", "star"):
            continue
        else:
            params.append(p.name)
    lines.append(f"{prefix}function {node.name}({', '.join(params)})")

    for stmt in node.body:
        emit_node(stmt, lines, indent + 1)
    lines.append(f"{prefix}end")


@register(IRClassDef)
def emit_class_def(node: IRClassDef, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    for dec in node.decorators:
        lines.append(f"{prefix}-- @{expr_to_str(dec.expr)}")
    lines.append(f"{prefix}-- class {node.name}")
    if node.bases:
        bases_str = ", ".join(expr_to_str(b) for b in node.bases)
        lines.append(f"{prefix}-- extends {bases_str}")
    lines.append(f"{prefix}{node.name} = {{}}")
    for stmt in node.body:
        emit_node(stmt, lines, indent)
    lines.append(f"{prefix}-- end class {node.name}")


@register(IRAssign)
def emit_assign(node: IRAssign, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    targets_str = ", ".join(expr_to_str(t) for t in node.targets)
    value_str = expr_to_str(node.value)
    op = ""
    if node.is_aug and node.aug_op:
        op = binop_to_lua(node.aug_op) + " "
    lines.append(f"{prefix}{targets_str} = {op}{value_str}")


@register(IRAnnotatedAssign)
def emit_annotated_assign(
    node: IRAnnotatedAssign, lines: list[str], indent: int
) -> None:
    prefix = _INDENT * indent
    ann_str = expr_to_str(node.annotation) if node.annotation else "?"
    target_str = expr_to_str(node.target)
    value_str = expr_to_str(node.value) if node.value else "nil"
    lines.append(f"{prefix}{target_str} = {value_str}  -- : {ann_str}")


@register(IRIf)
def emit_if(node: IRIf, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    test_str = expr_to_str(node.test)
    lines.append(f"{prefix}if {test_str} then")
    for stmt in node.body:
        emit_node(stmt, lines, indent + 1)

    if node.orelse:
        if len(node.orelse) == 1 and isinstance(node.orelse[0], IRIf):
            elif_node = node.orelse[0]
            test_str2 = expr_to_str(elif_node.test)
            lines.append(f"{prefix}elseif {test_str2} then")
            for stmt in elif_node.body:
                emit_node(stmt, lines, indent + 1)

        else:
            lines.append(f"{prefix}else")
            for stmt in node.orelse:
                emit_node(stmt, lines, indent + 1)

    lines.append(f"{prefix}end")


@register(IRWhile)
def emit_while(node: IRWhile, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    test_str = expr_to_str(node.test)
    lines.append(f"{prefix}while {test_str} do")
    for stmt in node.body:
        emit_node(stmt, lines, indent + 1)
    lines.append(f"{prefix}end")


@register(IRFor)
def emit_for(node: IRFor, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    target_str = expr_to_str(node.target)
    iter_str = expr_to_str(node.iter)
    lines.append(f"{prefix}for {target_str} in {iter_str} do")
    for stmt in node.body:
        emit_node(stmt, lines, indent + 1)
    lines.append(f"{prefix}end")


@register(IRReturn)
def emit_return(node: IRReturn, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    val = expr_to_str(node.value) if node.value else ""
    lines.append(f"{prefix}return {val}".rstrip())


@register(IRBreak)
def emit_break(node: IRBreak, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    lines.append(f"{prefix}break")


@register(IRContinue)
def emit_continue(node: IRContinue, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    lines.append(f"{prefix}-- continue (unsupported)")


@register(IRPass)
def emit_pass(node: IRPass, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    lines.append(f"{prefix}-- pass")


@register(IRExprStatement)
def emit_expr_stmt(node: IRExprStatement, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    lines.append(f"{prefix}{expr_to_str(node.expr)}")


@register(IRDelete)
def emit_delete(node: IRDelete, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    targets_str = ", ".join(expr_to_str(t) for t in node.targets)
    lines.append(f"{prefix}-- del {targets_str} (set to nil)")
    lines.append(f"{prefix}{targets_str} = nil")


@register(IRRaise)
def emit_raise(node: IRRaise, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    exc_str = expr_to_str(node.exc) if node.exc else ""
    cause_str = f" from {expr_to_str(node.cause)}" if node.cause else ""
    lines.append(f"{prefix}error({exc_str}{cause_str})")


@register(IRTry)
def emit_try(node: IRTry, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    lines.append(f"{prefix}-- try (using pcall/xpcall)")
    lines.append(f"{prefix}do")
    for stmt in node.body:
        emit_node(stmt, lines, indent + 1)
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
            emit_node(stmt, lines, indent + 1)
    if node.finalbody:
        lines.append(f"{prefix}-- finally:")
        for stmt in node.finalbody:
            emit_node(stmt, lines, indent + 1)


@register(IRWith)
def emit_with(node: IRWith, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    for item in node.items:
        ctx_str = expr_to_str(item.context_expr)
        if item.optional_vars:
            var_str = expr_to_str(item.optional_vars)
            lines.append(f"{prefix}-- with {ctx_str} as {var_str}:")
        else:
            lines.append(f"{prefix}-- with {ctx_str}:")
    for stmt in node.body:
        emit_node(stmt, lines, indent)


@register(IRComment)
def emit_comment(node: IRComment, lines: list[str], indent: int) -> None:
    prefix = _INDENT * indent
    lines.append(f"{prefix}-- {node.text}")


def expr_to_str(node: IRNode | None) -> str:
    """Возвращает строковое представление выражения (Lua синтаксис)."""
    if node is None:
        return "nil"
    if isinstance(node, IRSymbolRef):
        return f"SYM_{node.symbol_id}"
    if isinstance(node, IRConstant):
        return constant_to_str(node)
    if isinstance(node, (IRName)):
        return node.name
    if isinstance(node, IRAttribute):
        obj = expr_to_str(node.value)
        return f"{obj}.{node.attr}"
    if isinstance(node, IRSubscript):
        obj = expr_to_str(node.value)
        index = expr_to_str(node.index)
        return f"{obj}[{index}]"
    if isinstance(node, IRCall):
        func = expr_to_str(node.func)
        args = [expr_to_str(a) for a in node.args]
        kwargs = [f"{k}={expr_to_str(v)}" for k, v in node.kwargs.items()]
        all_args = args + kwargs
        return f"{func}({', '.join(all_args)})"
    if isinstance(node, IRUnaryOp):
        return unaryop_to_str(node)
    if isinstance(node, IRBinOp):
        return binop_to_str(node)
    if isinstance(node, IRIfExpr):
        return ifexpr_to_str(node)
    if isinstance(node, IRList):
        return "{" + ", ".join(expr_to_str(e) for e in node.elements) + "}"
    if isinstance(node, (IRTuple, IRSet)):
        return "{" + ", ".join(expr_to_str(e) for e in node.elements) + "}"
    if isinstance(node, IRDict):
        return dict_to_str(node)
    if isinstance(node, IRFString):
        return fstring_to_str(node)
    return f"<{type(node).__name__}>"


def constant_to_str(node: IRConstant) -> str:
    val = node.value
    if val is None:
        return "nil"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(val, (int, float)):
        return str(val)
    return repr(val)


def unaryop_to_str(node: IRUnaryOp) -> str:
    op = node.op
    operand = expr_to_str(node.operand)
    if op == "not":
        return f"not {operand}"
    elif op == "+":
        return f"+{operand}"
    elif op == "-":
        return f"-{operand}"
    elif op == "~":
        return f"~{operand}"
    return f"{op}({operand})"


def binop_to_str(node: IRBinOp) -> str:
    left = expr_to_str(node.left)
    right = expr_to_str(node.right)
    lua_op = binop_to_lua(node.op)
    return f"({left} {lua_op} {right})"


def ifexpr_to_str(node: IRIfExpr) -> str:
    test = expr_to_str(node.test)
    body = expr_to_str(node.body)
    orelse = expr_to_str(node.orelse)
    return f"({test} and {body} or {orelse})"


def dict_to_str(node: IRDict) -> str:
    items = []
    for item in node.items:
        k_str = _format_key(item.key)
        v_str = expr_to_str(item.value)
        items.append(f"{k_str} = {v_str}")
    return "{" + ", ".join(items) + "}"


def _format_key(key: IRNode) -> str:
    """Форматирует ключ словаря: если это имя и валидный идентификатор, без скобок."""
    if isinstance(key, IRName) and key.name.isidentifier():
        return key.name
    return f"[{expr_to_str(key)}]"


def fstring_to_str(node: IRFString) -> str:
    parts = []
    for part in node.parts:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, IRNode):
            parts.append(f"{{{expr_to_str(part)}}}")
        else:
            parts.append(str(part))
    return f'"{node.prefix}{"".join(parts)}"'


def binop_to_lua(op: IRBinOpType) -> str:
    mapping = {
        IRBinOpType.ADD: "+",
        IRBinOpType.SUB: "-",
        IRBinOpType.MUL: "*",
        IRBinOpType.DIV: "/",
        IRBinOpType.FLOORDIV: "//",
        IRBinOpType.MOD: "%",
        IRBinOpType.POW: "^",
        IRBinOpType.EQ: "==",
        IRBinOpType.NE: "~=",
        IRBinOpType.LT: "<",
        IRBinOpType.GT: ">",
        IRBinOpType.LE: "<=",
        IRBinOpType.GE: ">=",
        IRBinOpType.AND: "and",
        IRBinOpType.OR: "or",
        IRBinOpType.IN: "in",
        IRBinOpType.NOT_IN: "not in",
        IRBinOpType.IS: "==",
        IRBinOpType.IS_NOT: "~=",
        IRBinOpType.BIT_AND: "&",
        IRBinOpType.BIT_OR: "|",
        IRBinOpType.BIT_XOR: "~",
        IRBinOpType.LSHIFT: "<<",
        IRBinOpType.RSHIFT: ">>",
    }
    return mapping.get(op, f"<{op.name}>")


def dump_to_lua(ir_file: IRFile, symbol_table: SymbolTable | None = None) -> str:
    """Преобразует IRFile в строку с псевдо-Lua кодом."""
    lines = []

    if symbol_table is not None:
        lines.append("-- Symbols:")
        for sym in symbol_table.all():
            lines.append(f"--   {sym.id}: {sym.name} ({sym.kind})")
        lines.append("")

    for node in ir_file.body:
        emit_node(node, lines, 0)

    if ir_file.imports:
        import_lines = []
        for imp in ir_file.imports:
            if imp.is_from:
                names = ", ".join(
                    (alias if isinstance(item, tuple) else item)  # noqa: F821 # type: ignore
                    for item in imp.names
                )
                import_lines.append(f"-- from {'.'.join(imp.modules)} import {names}")
            else:
                for mod, name in zip(imp.modules, imp.names):
                    alias = name[1] if isinstance(name, tuple) else name
                    import_lines.append(f"-- import {mod} as {alias}")
        lines = import_lines + [""] + lines

    return "\n".join(lines)

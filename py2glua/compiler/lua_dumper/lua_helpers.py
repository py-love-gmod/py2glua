from __future__ import annotations

from plg_reader import (
    IRAttribute,
    IRBinOp,
    IRBinOpType,
    IRCall,
    IRConstant,
    IRDict,
    IRFString,
    IRIfExpr,
    IRList,
    IRName,
    IRNode,
    IRSet,
    IRSubscript,
    IRTuple,
    IRUnaryOp,
)

from ..analysis import IRSymbolRef


def expr_to_str(node: IRNode | None) -> str:
    if node is None:
        return "<FIXME None value>"

    if isinstance(node, IRSymbolRef):
        return f"SYM_{node.symbol_id}"

    if isinstance(node, IRConstant):
        return constant_to_str(node)

    if isinstance(node, IRName):
        return node.name

    if isinstance(node, IRAttribute):
        return f"{expr_to_str(node.value)}.{node.attr}"

    if isinstance(node, IRSubscript):
        return f"{expr_to_str(node.value)}[{expr_to_str(node.index)}]"

    if isinstance(node, IRCall):
        func = expr_to_str(node.func)
        args = [expr_to_str(a) for a in node.args]
        kwargs = [f"{k}={expr_to_str(v)}" for k, v in node.kwargs.items()]
        return f"{func}({', '.join(args + kwargs)})"

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
        return "<FIXME None value>"

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
    mapping = {"not": "not", "+": "+", "-": "-", "~": "~"}
    if op in mapping:
        return f"{mapping[op]}{operand}" if op != "not" else f"not {operand}"

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

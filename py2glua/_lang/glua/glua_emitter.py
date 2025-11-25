from __future__ import annotations

from .glua_ir_dataclass import (
    GluaAssign,
    GluaAttribute,
    GluaBinOp,
    GluaBinOpType,
    GluaBreak,
    GluaCall,
    GluaComment,
    GluaConstant,
    GluaContinue,
    GluaDoBlock,
    GluaFile,
    GluaForGeneric,
    GluaForNumeric,
    GluaFunctionDef,
    GluaIf,
    GluaIndex,
    GluaName,
    GluaNil,
    GluaNode,
    GluaReturn,
    GluaTable,
    GluaUnaryOp,
    GluaUnaryOpType,
    GluaVarArg,
    GluaWhile,
)


class GluaEmitter:
    def __init__(self, indent: str = "    "):
        self.indent = indent
        self.level = 0
        self.lines: list[str] = []

    # retion Low-level helpers
    def writeln(self, text: str = ""):
        self.lines.append(f"{self.indent * self.level}{text}")

    def emit(self, node: GluaNode | GluaFile) -> str:
        self.lines = []
        self.level = 0
        self.visit(node)
        return "\n".join(self.lines)

    def visit(self, node: GluaNode):
        name = node.__class__.__name__
        method = getattr(self, f"emit_{name}", None)
        if method is None:
            raise NotImplementedError(f"No emitter for node type: {name}")

        return method(node)

    def emit_block(self, body: list[GluaNode]):
        self.level += 1
        for stmt in body:
            self.visit(stmt)

        self.level -= 1

    # endregion

    # region File
    def emit_GluaFile(self, node: GluaFile):
        for stmt in node.body:
            self.visit(stmt)

    # endregion

    # region Basic expressions
    def expr(self, node: GluaNode) -> str:
        name = node.__class__.__name__
        method = getattr(self, f"expr_{name}", None)
        if method is None:
            raise NotImplementedError(f"No expr emitter for node: {name}")

        return method(node)

    def expr_GluaConstant(self, node: GluaConstant) -> str:
        v = node.value
        if v is None:
            return "{}"

        if isinstance(v, bool):
            return "true" if v else "false"

        if isinstance(v, str):
            return repr(v)

        return str(v)

    def expr_GluaNil(self, node: GluaNil) -> str:
        return "nil"

    def expr_GluaName(self, node: GluaName) -> str:
        return node.name

    def expr_GluaVarArg(self, node: GluaVarArg) -> str:
        return "..."

    def expr_GluaAttribute(self, node: GluaAttribute) -> str:
        return f"{self.expr(node.value)}.{node.attr}"

    def expr_GluaIndex(self, node: GluaIndex) -> str:
        return f"{self.expr(node.value)}[{self.expr(node.index)}]"

    # endregion

    # region Table
    def expr_GluaTable(self, node: GluaTable) -> str:
        parts = []
        for f in node.fields:
            if f.key is None:
                parts.append(self.expr(f.value))
            else:
                key = self.expr(f.key)
                parts.append(f"[{key}] = {self.expr(f.value)}")
        return "{ " + ", ".join(parts) + " }"

    # endregion

    # region Operators
    _bin_ops = {
        GluaBinOpType.OR: "or",
        GluaBinOpType.AND: "and",
        GluaBinOpType.EQ: "==",
        GluaBinOpType.NE: "~=",
        GluaBinOpType.LT: "<",
        GluaBinOpType.GT: ">",
        GluaBinOpType.LE: "<=",
        GluaBinOpType.GE: ">=",
        GluaBinOpType.CONCAT: "..",
        GluaBinOpType.ADD: "+",
        GluaBinOpType.SUB: "-",
        GluaBinOpType.MUL: "*",
        GluaBinOpType.DIV: "/",
        GluaBinOpType.MOD: "%",
        GluaBinOpType.POW: "^",
    }

    def expr_GluaBinOp(self, node: GluaBinOp) -> str:
        op = self._bin_ops[node.op]
        return f"({self.expr(node.left)} {op} {self.expr(node.right)})"

    _un_ops = {
        GluaUnaryOpType.MINUS: "-",
        GluaUnaryOpType.NOT: "not ",
        GluaUnaryOpType.LEN: "#",
    }

    def expr_GluaUnaryOp(self, node: GluaUnaryOp) -> str:
        op = self._un_ops[node.op]
        if node.op == GluaUnaryOpType.NOT:
            return f"(not {self.expr(node.value)})"

        return f"({op}{self.expr(node.value)})"

    # endregion

    # region Call
    def expr_GluaCall(self, node: GluaCall) -> str:
        args = ", ".join(self.expr(a) for a in node.args)
        return f"{self.expr(node.func)}({args})"

    def emit_GluaCall(self, node: GluaCall):
        self.writeln(self.expr_GluaCall(node))

    # endregion

    # region Assignment
    def emit_GluaAssign(self, node: GluaAssign):
        left = ", ".join(self.expr(t) for t in node.targets)

        if node.values:
            right = ", ".join(self.expr(v) for v in node.values)
            if node.is_local:
                self.writeln(f"local {left} = {right}")
            else:
                self.writeln(f"{left} = {right}")
        else:
            if node.is_local:
                self.writeln(f"local {left}")

            else:
                self.writeln(f"{left} = nil")

    # region If
    def emit_GluaIf(self, node: GluaIf):
        self.writeln(f"if {self.expr(node.test)} then")
        self.emit_block(node.body)

        if node.orelse:
            self.writeln("else")
            self.emit_block(node.orelse)

        self.writeln("end")

    # region While
    def emit_GluaWhile(self, node: GluaWhile):
        self.writeln(f"while {self.expr(node.test)} do")
        self.emit_block(node.body)
        self.writeln("end")

    # endregion

    # region ForNumeric
    def emit_GluaForNumeric(self, node: GluaForNumeric):
        var = node.var.name
        start = self.expr(node.start)
        stop = self.expr(node.stop)

        if node.step is None:
            self.writeln(f"for {var} = {start}, {stop} do")
        else:
            step = self.expr(node.step)
            self.writeln(f"for {var} = {start}, {stop}, {step} do")

        self.emit_block(node.body)
        self.writeln("end")

    # endregion

    # region ForGeneric
    def emit_GluaForGeneric(self, node: GluaForGeneric):
        targets = ", ".join(t.name for t in node.targets)
        iterator = self.expr(node.iter_expr)
        self.writeln(f"for {targets} in {iterator} do")
        self.emit_block(node.body)
        self.writeln("end")

    # endregion

    # region Return
    def emit_GluaReturn(self, node: GluaReturn):
        if node.values:
            vals = ", ".join(self.expr(v) for v in node.values)
            self.writeln(f"return {vals}")
        else:
            self.writeln("return")

    # region Break / Continue
    def emit_GluaBreak(self, node: GluaBreak):
        self.writeln("break")

    def emit_GluaContinue(self, node: GluaContinue):
        self.writeln("continue")

    # endregion

    # region Function
    def emit_GluaFunctionDef(self, node: GluaFunctionDef):
        params = []
        for p in node.params:
            if p.is_vararg:
                params.append("...")
            else:
                params.append(p.name)

        plist = ", ".join(params)

        if node.name is None:
            self.writeln(f"function({plist})")

        else:
            prefix = "local " if node.is_local else ""
            self.writeln(f"{prefix}function {node.name}({plist})")

        self.emit_block(node.body)
        self.writeln("end")

    # endregion

    # region Do-block / Comment
    def emit_GluaDoBlock(self, node: GluaDoBlock):
        self.writeln("do")
        self.emit_block(node.body)
        self.writeln("end")

    def emit_GluaComment(self, node: GluaComment):
        self.writeln(f"-- {node.text}")

    # endregion

from __future__ import annotations

from typing import List

from ..glua.glua_ir_dataclass import (
    GluaAssign,
    GluaAttribute,
    GluaBinOp,
    GluaBinOpType,
    GluaBreak,
    GluaCall,
    GluaComment,
    GluaConstant,
    GluaContinue,
    GluaFile,
    GluaForGeneric,
    GluaFuncParam,
    GluaFunctionDef,
    GluaIf,
    GluaIndex,
    GluaName,
    GluaNode,
    GluaReturn,
    GluaTable,
    GluaTableField,
    GluaUnaryOp,
    GluaUnaryOpType,
    GluaWhile,
)
from ..py.py_ir_dataclass import (
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
    PyIRDel,
    PyIRDict,
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
    PyIRTry,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyUnaryOPType,
)


class PyToGluaIR:
    # region Public API
    @classmethod
    def build_file(cls, py_file: PyIRFile) -> GluaFile:
        """Главная точка входа: PyIRFile -> GluaFile."""
        self = cls()
        return self._build_file(py_file)

    # endregion

    # region Core
    def _build_file(self, py_file: PyIRFile) -> GluaFile:
        glua_file = GluaFile(
            line=py_file.line,
            offset=py_file.offset,
            path=py_file.path,
            body=[],
        )

        realm = py_file.context.meta.get("realm") if py_file.context else None
        if realm is not None:
            glua_file.body.append(
                GluaComment(
                    line=py_file.line,
                    offset=py_file.offset,
                    text=f"Realm: {realm}",
                )
            )

        for node in py_file.body:
            self._emit_stmt(node, glua_file.body)

        return glua_file

    # endregion

    # region Statement dispatcher
    def _emit_stmt(self, node: PyIRNode, out: List[GluaNode]) -> None:
        if isinstance(node, PyIRFunctionDef):
            out.append(self._emit_funcdef(node))
            return

        if isinstance(node, PyIRClassDef):
            out.extend(self._emit_classdef(node))
            return

        if isinstance(node, PyIRIf):
            out.append(self._emit_if(node))
            return

        if isinstance(node, PyIRWhile):
            out.append(self._emit_while(node))
            return

        if isinstance(node, PyIRFor):
            out.append(self._emit_for(node))
            return

        if isinstance(node, PyIRReturn):
            out.append(self._emit_return(node))
            return

        if isinstance(node, PyIRBreak):
            out.append(GluaBreak(line=node.line, offset=node.offset))
            return

        if isinstance(node, PyIRContinue):
            out.append(GluaContinue(line=node.line, offset=node.offset))
            return

        if isinstance(node, PyIRAssign):
            out.append(self._emit_assign(node))
            return

        if isinstance(node, PyIRAugAssign):
            out.append(self._emit_augassign(node))
            return

        if isinstance(node, PyIRCall):
            out.append(self._emit_call_stmt(node))
            return

        if isinstance(node, PyIRComment):
            out.append(
                GluaComment(
                    line=node.line,
                    offset=node.offset,
                    text=node.value,
                )
            )
            return

        if isinstance(node, PyIRPass):
            # Просто коммент, чтобы не терять строку.
            out.append(
                GluaComment(
                    line=node.line,
                    offset=node.offset,
                    text="pass",
                )
            )
            return

        if isinstance(node, PyIRImport):
            # Пока просто коммент — импорты будут разруливаться выше уровнем.
            mods = ", ".join(node.modules)
            out.append(
                GluaComment(
                    line=node.line,
                    offset=node.offset,
                    text=f"import {mods}",
                )
            )
            return

        if isinstance(node, PyIRTry):
            out.extend(self._emit_try(node))
            return

        if isinstance(node, PyIRWith):
            out.extend(self._emit_with(node))
            return

        if isinstance(node, PyIRDel):
            out.append(
                GluaComment(
                    line=node.line,
                    offset=node.offset,
                    text="del (ignored in GLua)",
                )
            )
            return

        # Выражение как стейтмент (например, PyIRConstant / PyIRVarUse)
        expr = self._emit_expr(node)
        if isinstance(expr, GluaCall):
            out.append(expr)
        else:
            out.append(
                GluaComment(
                    line=node.line,
                    offset=node.offset,
                    text="expression statement (ignored)",
                )
            )

    # endregion

    # region Expressions

    def _emit_expr(self, node: PyIRNode) -> GluaNode:
        if isinstance(node, PyIRConstant):
            return GluaConstant(line=node.line, offset=node.offset, value=node.value)

        if isinstance(node, PyIRVarUse):
            return GluaName(line=node.line, offset=node.offset, name=node.name)

        if isinstance(node, PyIRAttribute):
            return GluaAttribute(
                line=node.line,
                offset=node.offset,
                value=self._emit_expr(node.value),
                attr=node.attr,
            )

        if isinstance(node, PyIRSubscript):
            return GluaIndex(
                line=node.line,
                offset=node.offset,
                value=self._emit_expr(node.value),
                index=self._emit_expr(node.index),
            )

        if isinstance(node, (PyIRList, PyIRTuple, PyIRSet)):
            # Все в табличный литерал { ... }
            fields: List[GluaTableField] = []
            elements = getattr(node, "elements", [])
            for el in elements:
                v = self._emit_expr(el)
                fields.append(
                    GluaTableField(
                        line=el.line,
                        offset=el.offset,
                        key=None,
                        value=v,
                    )
                )

            return GluaTable(
                line=node.line,
                offset=node.offset,
                fields=fields,
            )

        if isinstance(node, PyIRDict):
            fields: List[GluaTableField] = []
            for item in node.items:
                key = self._emit_expr(item.key)
                val = self._emit_expr(item.value)
                fields.append(
                    GluaTableField(
                        line=item.line,
                        offset=item.offset,
                        key=key,
                        value=val,
                    )
                )

            return GluaTable(
                line=node.line,
                offset=node.offset,
                fields=fields,
            )

        if isinstance(node, PyIRBinOP):
            return self._emit_binop(node)

        if isinstance(node, PyIRUnaryOP):
            return self._emit_unaryop(node)

        if isinstance(node, PyIRCall):
            return self._emit_call_expr(node)

        # Всё остальное пока не поддержано как expression
        raise NotImplementedError(
            f"Expression lowering not implemented for: {type(node).__name__}"
        )

    def _emit_binop(self, node: PyIRBinOP) -> GluaBinOp:
        mapping = {
            PyBinOPType.OR: GluaBinOpType.OR,
            PyBinOPType.AND: GluaBinOpType.AND,
            PyBinOPType.EQ: GluaBinOpType.EQ,
            PyBinOPType.NE: GluaBinOpType.NE,
            PyBinOPType.LT: GluaBinOpType.LT,
            PyBinOPType.GT: GluaBinOpType.GT,
            PyBinOPType.LE: GluaBinOpType.LE,
            PyBinOPType.GE: GluaBinOpType.GE,
            PyBinOPType.ADD: GluaBinOpType.ADD,
            PyBinOPType.SUB: GluaBinOpType.SUB,
            PyBinOPType.MUL: GluaBinOpType.MUL,
            PyBinOPType.DIV: GluaBinOpType.DIV,
            PyBinOPType.MOD: GluaBinOpType.MOD,
            PyBinOPType.POW: GluaBinOpType.POW,
        }

        if node.op not in mapping:
            raise NotImplementedError(
                f"Binary operator {node.op!r} is not supported in GLua lowering yet"
            )

        return GluaBinOp(
            line=node.line,
            offset=node.offset,
            op=mapping[node.op],
            left=self._emit_expr(node.left),
            right=self._emit_expr(node.right),
        )

    def _emit_unaryop(self, node: PyIRUnaryOP) -> GluaUnaryOp:
        mapping = {
            PyUnaryOPType.PLUS: GluaUnaryOpType.MINUS,  # унарный + игнорируем, оставляем value, но формально можно так
            PyUnaryOPType.MINUS: GluaUnaryOpType.MINUS,
            PyUnaryOPType.NOT: GluaUnaryOpType.NOT,
            # BIT_INV (~) пока не трогаем
        }

        if node.op not in mapping:
            raise NotImplementedError(
                f"Unary operator {node.op!r} is not supported in GLua lowering yet"
            )

        return GluaUnaryOp(
            line=node.line,
            offset=node.offset,
            op=mapping[node.op],
            value=self._emit_expr(node.value),
        )

    def _emit_call_expr(self, node: PyIRCall) -> GluaCall:
        if node.args_kw:
            raise NotImplementedError(
                "Keyword arguments are not supported in GLua lowering"
            )

        if not node.name:
            raise NotImplementedError(
                "Only simple function calls by name are supported"
            )

        func = GluaName(line=node.line, offset=node.offset, name=node.name)
        args = [self._emit_expr(a) for a in node.args_p]
        return GluaCall(
            line=node.line,
            offset=node.offset,
            func=func,
            args=args,
        )

    def _emit_call_stmt(self, node: PyIRCall) -> GluaCall:
        return self._emit_call_expr(node)

    # endregion

    # region Assignments
    def _flatten_targets(self, targets: List[PyIRNode]) -> List[PyIRNode]:
        flat: List[PyIRNode] = []
        for t in targets:
            if isinstance(t, PyIRTuple):
                flat.extend(t.elements)
            else:
                flat.append(t)

        return flat

    def _emit_assign(self, node: PyIRAssign) -> GluaAssign:
        targets = self._flatten_targets(node.targets)
        glua_targets: List[GluaNode] = [self._emit_expr(t) for t in targets]
        value = self._emit_expr(node.value)

        values: List[GluaNode]
        if isinstance(node.value, PyIRTuple):
            values = [self._emit_expr(el) for el in node.value.elements]

        else:
            values = [value]

        return GluaAssign(
            line=node.line,
            offset=node.offset,
            targets=glua_targets,
            values=values,
            is_local=False,
        )

    def _emit_augassign(self, node: PyIRAugAssign) -> GluaAssign:
        op_map = {
            PyAugAssignType.ADD: PyBinOPType.ADD,
            PyAugAssignType.SUB: PyBinOPType.SUB,
            PyAugAssignType.MUL: PyBinOPType.MUL,
            PyAugAssignType.DIV: PyBinOPType.DIV,
            PyAugAssignType.MOD: PyBinOPType.MOD,
            PyAugAssignType.POW: PyBinOPType.POW,
        }
        if node.op not in op_map:
            raise NotImplementedError(
                f"Augmented assign {node.op!r} is not supported in GLua lowering yet"
            )

        binop = PyIRBinOP(
            line=node.line,
            offset=node.offset,
            op=op_map[node.op],
            left=node.target,
            right=node.value,
        )
        value = self._emit_binop(binop)

        return GluaAssign(
            line=node.line,
            offset=node.offset,
            targets=[self._emit_expr(node.target)],
            values=[value],
            is_local=False,
        )

    # endregion

    # region Control Flow
    def _emit_if(self, node: PyIRIf) -> GluaIf:
        test = self._emit_expr(node.test)
        body: List[GluaNode] = []
        for stmt in node.body:
            self._emit_stmt(stmt, body)

        orelse: List[GluaNode] = []
        for stmt in node.orelse:
            self._emit_stmt(stmt, orelse)

        return GluaIf(
            line=node.line,
            offset=node.offset,
            test=test,
            body=body,
            orelse=orelse,
        )

    def _emit_while(self, node: PyIRWhile) -> GluaWhile:
        test = self._emit_expr(node.test)
        body: List[GluaNode] = []
        for stmt in node.body:
            self._emit_stmt(stmt, body)

        return GluaWhile(
            line=node.line,
            offset=node.offset,
            test=test,
            body=body,
        )

    def _emit_for(self, node: PyIRFor) -> GluaNode:
        targets: List[GluaName] = []

        if isinstance(node.target, PyIRVarUse):
            targets.append(
                GluaName(
                    line=node.target.line,
                    offset=node.target.offset,
                    name=node.target.name,
                )
            )

        elif isinstance(node.target, PyIRTuple):
            for el in node.target.elements:
                if not isinstance(el, PyIRVarUse):
                    raise NotImplementedError("Complex 'for' targets are not supported")
                targets.append(
                    GluaName(
                        line=el.line,
                        offset=el.offset,
                        name=el.name,
                    )
                )

        else:
            raise NotImplementedError(
                "Only simple names or tuple of names supported as 'for' target"
            )

        iter_expr = self._emit_expr(node.iter)
        body: List[GluaNode] = []
        for stmt in node.body:
            self._emit_stmt(stmt, body)

        return GluaForGeneric(
            line=node.line,
            offset=node.offset,
            targets=targets,
            iter_expr=iter_expr,
            body=body,
        )

    def _emit_return(self, node: PyIRReturn) -> GluaReturn:
        value = self._emit_expr(node.value) if node.value is not None else None
        values: List[GluaNode] = []
        if value is not None:
            if isinstance(node.value, PyIRTuple):
                values = [self._emit_expr(el) for el in node.value.elements]

            else:
                values = [value]

        return GluaReturn(
            line=node.line,
            offset=node.offset,
            values=values,
        )

    # endregion

    # region Functions / Classes / Try / With
    def _emit_funcdef(self, node: PyIRFunctionDef) -> GluaFunctionDef:
        params: List[GluaFuncParam] = []

        return GluaFunctionDef(
            line=node.line,
            offset=node.offset,
            name=node.name,
            params=params,
            body=self._emit_block(node.body),
            is_local=False,
        )

    def _emit_classdef(self, node: PyIRClassDef) -> List[GluaNode]:
        out: List[GluaNode] = []
        out.append(
            GluaComment(
                line=node.line,
                offset=node.offset,
                text=f"class {node.name} (lowered to table placeholder)",
            )
        )

        out.append(
            GluaAssign(
                line=node.line,
                offset=node.offset,
                targets=[GluaName(line=node.line, offset=node.offset, name=node.name)],
                values=[
                    GluaTable(
                        line=node.line,
                        offset=node.offset,
                        fields=[],
                    )
                ],
                is_local=False,
            )
        )

        return out

    def _emit_block(self, body: List[PyIRNode]) -> List[GluaNode]:
        result: List[GluaNode] = []
        for stmt in body:
            self._emit_stmt(stmt, result)
        return result

    def _emit_try(self, node: PyIRTry) -> List[GluaNode]:
        out: List[GluaNode] = []
        out.append(
            GluaComment(
                line=node.line,
                offset=node.offset,
                text="try/except/finally block (semantics not preserved)",
            )
        )

        for stmt in node.body:
            self._emit_stmt(stmt, out)

        for h in node.handlers:
            out.append(
                GluaComment(
                    line=h.line,
                    offset=h.offset,
                    text="except handler body inlined",
                )
            )
            for stmt in h.body:
                self._emit_stmt(stmt, out)

        for stmt in node.orelse:
            self._emit_stmt(stmt, out)

        for stmt in node.finalbody:
            self._emit_stmt(stmt, out)

        return out

    def _emit_with(self, node: PyIRWith) -> List[GluaNode]:
        out: List[GluaNode] = []
        header_desc: List[str] = []
        for item in node.items:
            ctx = self._emit_expr(item.context_expr)
            header_desc.append(f"(with {type(ctx).__name__})")

        out.append(
            GluaComment(
                line=node.line,
                offset=node.offset,
                text="with-statement (context manager semantics not preserved)",
            )
        )
        for stmt in node.body:
            self._emit_stmt(stmt, out)

        return out

    # endregion

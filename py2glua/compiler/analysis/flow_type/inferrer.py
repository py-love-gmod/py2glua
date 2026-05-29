from __future__ import annotations

from typing import Dict, List, Optional

from plg_reader import (
    IRAnnotatedAssign,
    IRAssign,
    IRAttribute,
    IRBinOp,
    IRBinOpType,
    IRCall,
    IRClassDef,
    IRConstant,
    IRDelete,
    IRDict,
    IRExprStatement,
    IRFile,
    IRFor,
    IRFString,
    IRFStringDebug,
    IRFunctionDef,
    IRIf,
    IRIfExpr,
    IRList,
    IRName,
    IRNode,
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

from ..symtable import IRSymbolRef, Scope, SymbolTable
from .annotation_resolver import AnnotationResolver, _constant_type
from .builtins import get_binop_result, get_iteration_element_type
from .env import TypeEnv
from .result import InferenceResult
from .types import (
    AnyType,
    BoolType,
    CallableType,
    ClassType,
    DictType,
    ListType,
    NoneType,
    SetType,
    StrType,
    TupleType,
    Type,
    UnionType,
)


class TypeInferrer:
    _dispatch: dict[type, str] = {
        IRFunctionDef: "_func_def",
        IRClassDef: "_class_def",
        IRAssign: "_assign",
        IRAnnotatedAssign: "_ann_assign",
        IRReturn: "_return",
        IRExprStatement: "_expr_stmt",
        IRIf: "_if",
        IRWhile: "_while",
        IRFor: "_for",
        IRWith: "_with",
        IRTry: "_try",
        IRRaise: "_raise",
        IRDelete: "_delete",
    }

    def __init__(
        self,
        symbol_table: SymbolTable,
        external_tables: Dict[str, SymbolTable] | None = None,
    ):
        self.symbol_table = symbol_table
        self.external_tables = external_tables or {}
        self.env = TypeEnv()
        self.result = InferenceResult()
        self.current_function: Optional[CallableType] = None
        self._first_pass: bool = True

    def infer_file(self, ir_file: IRFile) -> InferenceResult:
        self._first_pass = True
        for stmt in ir_file.body:
            self._infer_stmt(stmt)

        self._first_pass = False

        for stmt in ir_file.body:
            self._infer_stmt(stmt)

        return self.result

    def _infer_stmt(self, node: IRNode):
        handler_name = self._dispatch.get(type(node))
        if handler_name is not None:
            handler = getattr(self, handler_name)
            handler(node)

    def _func_def(self, node: IRFunctionDef):
        resolver = AnnotationResolver(self.symbol_table, Scope())
        ptypes = [resolver.resolve(p.annotation) for p in node.params]
        rtype = resolver.resolve(node.returns)
        ft = CallableType(tuple(ptypes), rtype)
        self._set_type(node, ft)
        self.env.set(node.name, ft)

        if self._first_pass:
            return

        old_env, old_ret = self.env, self.current_function
        self.env = self.env.push()
        for p, t in zip(node.params, ptypes):
            self.env.set(p.name, t)

        self.current_function = ft
        for s in node.body:
            self._infer_stmt(s)

        self.env = old_env
        self.current_function = old_ret

    def _class_def(self, node: IRClassDef):
        fields: dict[str, Type] = {}
        for stmt in node.body:
            if isinstance(stmt, IRFunctionDef):
                resolver = AnnotationResolver(self.symbol_table, Scope())
                ptypes = [resolver.resolve(p.annotation) for p in stmt.params]
                rtype = resolver.resolve(stmt.returns)
                method_type = CallableType(tuple(ptypes), rtype)
                fields[stmt.name] = method_type

            elif isinstance(stmt, IRAnnotatedAssign) and isinstance(
                stmt.target, (IRName, IRSymbolRef)
            ):
                resolver = AnnotationResolver(self.symbol_table, Scope())
                ft = resolver.resolve(stmt.annotation)
                fields[stmt.target.name] = ft

        base_class: Optional[ClassType] = None
        for b in node.bases:
            bt = self._expr_type(b)
            if isinstance(bt, ClassType):
                base_class = bt
                break

        ct = ClassType(node.name, fields, base=base_class)
        self._set_type(node, ct)
        self.env.set(node.name, ct)

        if self._first_pass:
            return

        for stmt in node.body:
            if isinstance(stmt, IRFunctionDef):
                method_type = fields[stmt.name]
                if isinstance(method_type, CallableType):
                    old_env, old_ret = self.env, self.current_function
                    self.env = self.env.push()
                    for p, t in zip(stmt.params, method_type.params):
                        self.env.set(p.name, t)

                    self.current_function = method_type
                    for s in stmt.body:
                        self._infer_stmt(s)

                    self.env = old_env
                    self.current_function = old_ret

            elif isinstance(stmt, IRAnnotatedAssign):
                self._ann_assign(stmt)

    def _assign(self, node: IRAssign):
        if self._first_pass:
            return

        vt = self._expr_type(node.value)
        for t in node.targets:
            self._assign_target(t, vt)

    def _ann_assign(self, node: IRAnnotatedAssign):
        resolver = AnnotationResolver(self.symbol_table, Scope())
        at = resolver.resolve(node.annotation)
        self._set_type(node, at)

        if isinstance(node.target, (IRName, IRSymbolRef)):
            self.env.set(node.target.name, at)

        if self._first_pass:
            return

        if node.value:
            vt = self._expr_type(node.value)
            if not vt.is_subtype(at):
                self._error(
                    f"Несоответствие типа при присваивании: ожидался {at}, получен {vt}",
                    node,
                )

    def _return(self, node: IRReturn):
        if self._first_pass:
            return

        if self.current_function is None:
            self._error("return вне тела функции", node)
            return

        expected = self.current_function.return_type
        actual = self._expr_type(node.value) if node.value else NoneType
        if not actual.is_subtype(expected):
            self._error(
                f"Несоответствие типа возвращаемого значения: ожидался {expected}, получен {actual}",
                node,
            )

    def _if(self, node: IRIf):
        if self._first_pass:
            return

        self._expr_type(node.test)
        for s in node.body:
            self._infer_stmt(s)

        for s in node.orelse:
            self._infer_stmt(s)

    def _while(self, node: IRWhile):
        if self._first_pass:
            return

        self._expr_type(node.test)
        for s in node.body:
            self._infer_stmt(s)

    def _for(self, node: IRFor):
        if self._first_pass:
            return

        iter_type = self._expr_type(node.iter)
        elem_type = get_iteration_element_type(iter_type)
        self._assign_target(node.target, elem_type)
        for s in node.body:
            self._infer_stmt(s)

    def _with(self, node: IRWith):
        if self._first_pass:
            return

        for item in node.items:
            self._expr_type(item.context_expr)
            if item.optional_vars:
                self._assign_target(item.optional_vars, AnyType)

        for s in node.body:
            self._infer_stmt(s)

    def _try(self, node: IRTry):
        if self._first_pass:
            return

        for s in node.body:
            self._infer_stmt(s)

        for h in node.handlers:
            if h.type:
                self._expr_type(h.type)

            if h.name:
                self.env.set(h.name, AnyType)

            for s in h.body:
                self._infer_stmt(s)

        for s in node.orelse:
            self._infer_stmt(s)

        for s in node.finalbody:
            self._infer_stmt(s)

    def _expr_stmt(self, node: IRExprStatement):
        if self._first_pass:
            return

        self._expr_type(node.expr)

    def _raise(self, node: IRRaise):
        if self._first_pass:
            return

        if node.exc:
            self._expr_type(node.exc)

    def _delete(self, node: IRDelete):
        if self._first_pass:
            return

        for t in node.targets:
            self._expr_type(t)

    def _expr_type(self, node: Optional[IRNode]) -> Type:
        if node is None:
            return AnyType

        nid = id(node)
        if nid in self.result.node_types:
            return self.result.node_types[nid]

        t = self._compute_expr(node)
        self.result.node_types[nid] = t
        return t

    def _compute_expr(self, node: IRNode) -> Type:
        if isinstance(node, IRConstant):
            return _constant_type(node)

        if isinstance(node, (IRName, IRSymbolRef)):
            envt = self.env.get(node.name)
            if envt is not AnyType or node.name in self.env._vars:
                return envt

            sym = self._find_sym(node.name)
            if sym and sym.definition:
                if isinstance(sym.definition, IRFunctionDef):
                    return self._node_type(sym.definition) or AnyType

                if isinstance(sym.definition, IRClassDef):
                    return ClassType(sym.definition.name)

            return AnyType

        if isinstance(node, IRAttribute):
            obj = self._expr_type(node.value)
            if isinstance(obj, ClassType):
                field_type = obj.get_field(node.attr)
                if field_type is AnyType:
                    self._warning(
                        f"Доступ к неизвестному атрибуту '{node.attr}' класса '{obj.name}'",
                        node,
                    )
                return field_type

            if obj is AnyType:
                self._warning(
                    f"Доступ к атрибуту '{node.attr}' у значения с неизвестным типом",
                    node,
                )
            return AnyType

        if isinstance(node, IRSubscript):
            base = self._expr_type(node.value)
            if isinstance(base, ListType):
                return base.item

            if isinstance(base, DictType):
                return base.value

            if (
                isinstance(base, TupleType)
                and isinstance(node.index, IRConstant)
                and isinstance(node.index.value, int)
            ):
                i = node.index.value
                if 0 <= i < len(base.items):
                    return base.items[i]

            return AnyType

        if isinstance(node, IRCall):
            ft = self._expr_type(node.func)
            if isinstance(ft, CallableType):
                self._check_call(node, ft)
                return ft.return_type

            if isinstance(ft, ClassType):
                return ft

            if isinstance(node.func, (IRName, IRSymbolRef)):
                self._warning(
                    f"Вызов неизвестной функции '{node.func.name}' - возможен nil",
                    node,
                )

            elif isinstance(node.func, IRAttribute):
                self._warning(
                    "Вызов неизвестного метода - возможен nil",
                    node,
                )

            else:
                self._warning(
                    "Вызов выражения с неизвестным типом - возможен nil", node
                )

            return AnyType

        if isinstance(node, IRUnaryOp):
            return self._expr_type(node.operand)

        if isinstance(node, IRBinOp):
            return self._binop(node)

        if isinstance(node, IRIfExpr):
            t1 = self._expr_type(node.body)
            t2 = self._expr_type(node.orelse)
            return t1 if t1 == t2 else UnionType((t1, t2))

        if isinstance(node, IRList):
            elem_types = [self._expr_type(e) for e in node.elements]
            return ListType(self._common(elem_types))

        if isinstance(node, IRTuple):
            return TupleType(tuple(self._expr_type(e) for e in node.elements))

        if isinstance(node, IRSet):
            elem_types = [self._expr_type(e) for e in node.elements]
            return SetType(self._common(elem_types))

        if isinstance(node, IRDict):
            keys = [item.key for item in node.items]
            values = [item.value for item in node.items]
            return DictType(
                self._common([self._expr_type(k) for k in keys]),
                self._common([self._expr_type(v) for v in values]),
            )

        if isinstance(node, (IRFString, IRFStringDebug)):
            return StrType

        return AnyType

    def _binop(self, node: IRBinOp) -> Type:
        left = self._expr_type(node.left)
        right = self._expr_type(node.right)
        result = get_binop_result(left, node.op, right)
        if result is not None:
            return result

        if node.op in (
            IRBinOpType.EQ,
            IRBinOpType.NE,
            IRBinOpType.LT,
            IRBinOpType.LE,
            IRBinOpType.GT,
            IRBinOpType.GE,
            IRBinOpType.IN,
            IRBinOpType.NOT_IN,
        ):
            return BoolType
        if node.op in (IRBinOpType.AND, IRBinOpType.OR):
            return self._common([left, right])
        return AnyType

    def _common(self, types: List[Type]) -> Type:
        if not types:
            return AnyType

        first = types[0]
        return first if all(t == first for t in types) else UnionType(tuple(types))

    def _check_call(self, node: IRCall, ft: CallableType):
        params = ft.params
        param_names: list[str] = []
        if isinstance(node.func, IRSymbolRef):
            sym = self._find_sym(node.func.name)
            if sym and isinstance(sym.definition, IRFunctionDef):
                param_names = [p.name for p in sym.definition.params]

        for i, arg in enumerate(node.args):
            if i < len(params):
                arg_t = self._expr_type(arg)
                if not arg_t.is_subtype(params[i]):
                    self._error(
                        f"Несоответствие типа аргумента {i}: ожидался {params[i]}, получен {arg_t}",
                        node,
                    )
            else:
                self._error("Слишком много позиционных аргументов при вызове", node)

        for kw_name, kw_value in node.kwargs.items():
            if kw_name in param_names:
                idx = param_names.index(kw_name)
                if idx < len(params):
                    kw_t = self._expr_type(kw_value)
                    if not kw_t.is_subtype(params[idx]):
                        self._error(
                            f"Несоответствие типа именованного аргумента '{kw_name}': ожидался {params[idx]}, получен {kw_t}",
                            node,
                        )

                else:
                    self._error(
                        f"Именованный аргумент '{kw_name}' вне допустимого диапазона",
                        node,
                    )

            else:
                self._error(f"Неожиданный именованный аргумент '{kw_name}'", node)

    def _assign_target(self, target: IRNode, val_type: Type):
        if isinstance(target, (IRName, IRSymbolRef)):
            existing = self.env.get(target.name)
            if existing is AnyType:
                self.env.set(target.name, val_type)

            elif not val_type.is_subtype(existing):
                self._error(
                    f"Несоответствие типа при присваивании переменной '{target.name}': ожидался {existing}, получен {val_type}",
                    target,
                )

        elif isinstance(target, (IRTuple, IRList)):
            if isinstance(val_type, TupleType) and len(val_type.items) == len(
                target.elements
            ):
                for e, t in zip(target.elements, val_type.items):
                    self._assign_target(e, t)

            else:
                for e in target.elements:
                    self._assign_target(e, AnyType)

        self._expr_type(target)

    def _find_sym(self, name: str):
        for s in self.symbol_table.all():
            if s.name == name:
                return s

        return None

    def _node_type(self, node: IRNode) -> Optional[Type]:
        return self.result.node_types.get(id(node))

    def _set_type(self, node: IRNode, typ: Type):
        self.result.node_types[id(node)] = typ

    def _error(self, msg: str, node: IRNode):
        pos = getattr(node, "pos", (0, 0))
        self.result.errors.append(f"{pos}: {msg}")

    def _warning(self, msg: str, node: IRNode):
        pos = getattr(node, "pos", (0, 0))
        self.result.warnings.append(f"{pos}: {msg}")

# FIXME
from __future__ import annotations

from typing import Any, cast

from plg_reader import (
    IRAnnotatedAssign,
    IRAssign,
    IRAttribute,
    IRClassDef,
    IRDecorator,
    IRDelete,
    IRExceptHandler,
    IRExprStatement,
    IRFile,
    IRFor,
    IRFunctionDef,
    IRIf,
    IRImport,
    IRList,
    IRName,
    IRNode,
    IRParam,
    IRRaise,
    IRReturn,
    IRSubscript,
    IRTry,
    IRTuple,
    IRWhile,
    IRWith,
    IRWithItem,
)

from .symbols import IRSymbolRef, Scope, SymbolTable


class LocalSymbolResolver:
    @classmethod
    def resolve(
        cls, ir_file: IRFile, external_scopes: dict[str, Scope] | None = None
    ) -> tuple[IRFile, SymbolTable]:
        resolver = cls(external_scopes)
        return resolver._resolve(ir_file)

    def __init__(self, external_scopes: dict[str, Scope] | None = None):
        self.external_scopes = external_scopes or {}
        self.symbol_table = SymbolTable()
        self._scope_map: dict[int, Scope] = {}

    def _resolve(self, ir_file: IRFile) -> tuple[IRFile, SymbolTable]:
        root = Scope()
        self._collect_imports(ir_file.imports, root)
        self._collect_definitions(ir_file.body, root)
        new_body = self._transform_statements(ir_file.body, root)
        new_ir = IRFile(
            path=ir_file.path, body=new_body, imports=ir_file.imports, pos=ir_file.pos
        )
        return new_ir, self.symbol_table

    def _collect_imports(self, imports: list[IRImport], scope: Scope):
        for imp in imports:
            if imp.is_from:
                for name_item in imp.names:
                    alias = name_item[1] if isinstance(name_item, tuple) else name_item
                    self._add_symbol(alias, "import", scope)
            else:
                for mod_path, name_item in zip(imp.modules, imp.names):
                    alias = name_item[1] if isinstance(name_item, tuple) else name_item
                    parts = mod_path.split(".")
                    for i in range(1, len(parts) + 1):
                        self._add_symbol(".".join(parts[:i]), "module", scope)
                    if alias != mod_path:
                        self._add_symbol(alias, "module", scope)

    def _collect_definitions(self, stmts: list[IRNode], scope: Scope):
        for stmt in stmts:
            self._collect_node(stmt, scope)

    def _collect_node(self, node: IRNode, scope: Scope):
        if isinstance(node, IRFunctionDef):
            self._add_symbol(node.name, "function", scope, node)
            func_scope = Scope(parent=scope)
            self._scope_map[id(node)] = func_scope
            for p in node.params:
                if p.name:
                    self._add_symbol(p.name, "param", func_scope, p)
            self._collect_definitions(node.body, func_scope)
        elif isinstance(node, IRClassDef):
            self._add_symbol(node.name, "class", scope, node)
            class_scope = Scope(parent=scope)
            self._scope_map[id(node)] = class_scope
            self._collect_definitions(node.body, class_scope)
        elif isinstance(node, IRAssign):
            for t in node.targets:
                self._collect_target_names(t, scope)
        elif isinstance(node, IRAnnotatedAssign):
            if isinstance(node.target, IRName):
                self._add_symbol(node.target.name, "variable", scope, node.target)
        elif isinstance(node, IRFor):
            self._collect_target_names(node.target, scope)
            self._collect_definitions(node.body, scope)
        elif isinstance(node, IRWith):
            for item in node.items:
                if item.optional_vars:
                    self._collect_target_names(item.optional_vars, scope)
            self._collect_definitions(node.body, scope)
        elif isinstance(node, IRTry):
            self._collect_definitions(node.body, scope)
            for h in node.handlers:
                if h.name:
                    self._add_symbol(h.name, "variable", scope, h)
                self._collect_definitions(h.body, scope)
            self._collect_definitions(node.orelse, scope)
            self._collect_definitions(node.finalbody, scope)
        elif isinstance(node, IRIf):
            self._collect_definitions(node.body, scope)
            if node.orelse:
                self._collect_definitions(node.orelse, scope)
        elif isinstance(node, IRWhile):
            self._collect_definitions(node.body, scope)
        else:
            self._walk_node_fields(node, scope)

    def _collect_target_names(self, target: IRNode, scope: Scope):
        if isinstance(target, IRName):
            self._add_symbol(target.name, "variable", scope, target)
        elif isinstance(target, (IRTuple, IRList)):
            for el in target.elements:
                self._collect_target_names(el, scope)

    def _add_symbol(
        self, name: str, kind: str, scope: Scope, definition: IRNode | None = None
    ):
        if name not in scope.symbols:
            sym = self.symbol_table.create(name, kind, definition)
            scope.symbols[name] = (sym.id, kind)

    def _walk_node_fields(self, node: IRNode, scope: Scope):
        for fld in node.__dataclass_fields__:
            if fld == "pos":
                continue
            val = getattr(node, fld)
            if isinstance(val, IRNode):
                self._collect_node(val, scope)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, IRNode):
                        self._collect_node(item, scope)
            elif isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, IRNode):
                        self._collect_node(v, scope)

    def _transform_statements(self, stmts: list[IRNode], scope: Scope) -> list[IRNode]:
        return [self._transform_node(stmt, scope) for stmt in stmts]

    def _transform_node(self, node: IRNode, scope: Scope) -> IRNode:
        if isinstance(node, IRName):
            return self._resolve_name_to_ref(node.name, node.pos, scope, is_write=False)

        if isinstance(node, IRFunctionDef):
            func_scope = self._scope_map[id(node)]
            new_params = [self._transform_param(p, scope) for p in node.params]
            new_body = self._transform_statements(node.body, func_scope)
            new_returns = (
                self._transform_expr(node.returns, scope) if node.returns else None
            )
            new_decors = [
                IRDecorator(pos=d.pos, expr=self._transform_expr(d.expr, scope))
                for d in node.decorators
            ]
            return IRFunctionDef(
                pos=node.pos,
                name=node.name,
                params=new_params,
                returns=new_returns,
                body=new_body,
                decorators=new_decors,
            )

        elif isinstance(node, IRClassDef):
            class_scope = self._scope_map[id(node)]
            new_bases = [self._transform_expr(b, scope) for b in node.bases]
            new_body = self._transform_statements(node.body, class_scope)
            new_decors = [
                IRDecorator(pos=d.pos, expr=self._transform_expr(d.expr, scope))
                for d in node.decorators
            ]
            return IRClassDef(
                pos=node.pos,
                name=node.name,
                bases=new_bases,
                body=new_body,
                decorators=new_decors,
            )

        elif isinstance(node, IRAssign):
            new_targets = [self._transform_target(t, scope) for t in node.targets]
            new_value = self._transform_expr(node.value, scope)
            return IRAssign(
                pos=node.pos,
                targets=new_targets,
                value=new_value,
                is_aug=node.is_aug,
                aug_op=node.aug_op,
            )

        elif isinstance(node, IRAnnotatedAssign):
            new_target = self._transform_target(node.target, scope)
            new_ann = (
                self._transform_expr(node.annotation, scope)
                if node.annotation
                else None
            )
            new_val = self._transform_expr(node.value, scope) if node.value else None
            return IRAnnotatedAssign(
                pos=node.pos, target=new_target, annotation=new_ann, value=new_val
            )

        elif isinstance(node, IRFor):
            new_target = self._transform_target(node.target, scope)
            new_iter = self._transform_expr(node.iter, scope)
            new_body = self._transform_statements(node.body, scope)
            return IRFor(pos=node.pos, target=new_target, iter=new_iter, body=new_body)

        elif isinstance(node, IRWith):
            new_items = [
                IRWithItem(
                    pos=item.pos,
                    context_expr=self._transform_expr(item.context_expr, scope),
                    optional_vars=self._transform_target(item.optional_vars, scope)
                    if item.optional_vars
                    else None,
                )
                for item in node.items
            ]
            new_body = self._transform_statements(node.body, scope)
            return IRWith(pos=node.pos, items=new_items, body=new_body)

        elif isinstance(node, IRIf):
            new_test = self._transform_expr(node.test, scope)
            new_body = self._transform_statements(node.body, scope)
            new_orelse = self._transform_statements(node.orelse, scope)
            return IRIf(pos=node.pos, test=new_test, body=new_body, orelse=new_orelse)

        elif isinstance(node, IRWhile):
            new_test = self._transform_expr(node.test, scope)
            new_body = self._transform_statements(node.body, scope)
            return IRWhile(pos=node.pos, test=new_test, body=new_body)

        elif isinstance(node, IRExprStatement):
            return IRExprStatement(
                pos=node.pos, expr=self._transform_expr(node.expr, scope)
            )

        elif isinstance(node, IRReturn):
            new_val = self._transform_expr(node.value, scope) if node.value else None
            return IRReturn(pos=node.pos, value=new_val)

        elif isinstance(node, IRDelete):
            new_targets = [self._transform_expr(t, scope) for t in node.targets]
            return IRDelete(pos=node.pos, targets=new_targets)

        elif isinstance(node, IRRaise):
            new_exc = self._transform_expr(node.exc, scope)
            new_cause = self._transform_expr(node.cause, scope)
            return IRRaise(pos=node.pos, exc=new_exc, cause=new_cause)

        elif isinstance(node, IRTry):
            new_body = self._transform_statements(node.body, scope)
            new_handlers = [
                IRExceptHandler(
                    pos=h.pos,
                    type=self._transform_expr(h.type, scope) if h.type else None,
                    name=h.name,
                    body=self._transform_statements(h.body, scope),
                )
                for h in node.handlers
            ]
            new_orelse = self._transform_statements(node.orelse, scope)
            new_finalbody = self._transform_statements(node.finalbody, scope)
            return IRTry(
                pos=node.pos,
                body=new_body,
                handlers=new_handlers,
                orelse=new_orelse,
                finalbody=new_finalbody,
            )

        else:
            kwargs: dict[str, Any] = {"pos": node.pos}
            for fld in node.__dataclass_fields__:
                if fld == "pos":
                    continue
                val = getattr(node, fld)
                if isinstance(val, IRNode):
                    kwargs[fld] = self._transform_node(val, scope)
                elif isinstance(val, list):
                    kwargs[fld] = [
                        self._transform_node(cast(IRNode, v), scope)
                        if isinstance(v, IRNode)
                        else v
                        for v in val
                    ]
                elif isinstance(val, dict):
                    kwargs[fld] = {
                        k: self._transform_node(cast(IRNode, v), scope)
                        if isinstance(v, IRNode)
                        else v
                        for k, v in val.items()
                    }
                else:
                    kwargs[fld] = val
            return type(node)(**kwargs)

    def _transform_param(self, param: IRParam, scope: Scope) -> IRParam:
        new_ann = (
            self._transform_expr(param.annotation, scope) if param.annotation else None
        )
        new_default = (
            self._transform_expr(param.default, scope) if param.default else None
        )
        return IRParam(
            pos=param.pos,
            name=param.name,
            kind=param.kind,
            annotation=new_ann,
            default=new_default,
        )

    def _transform_target(self, node: IRNode, scope: Scope) -> IRNode:
        if isinstance(node, IRName):
            return self._resolve_name_to_ref(node.name, node.pos, scope, is_write=True)
        elif isinstance(node, IRAttribute):
            new_val = self._transform_expr(node.value, scope)
            return IRAttribute(pos=node.pos, value=new_val, attr=node.attr)
        elif isinstance(node, IRSubscript):
            return IRSubscript(
                pos=node.pos,
                value=self._transform_expr(node.value, scope),
                index=self._transform_expr(node.index, scope),
            )
        elif isinstance(node, (IRTuple, IRList)):
            return type(node)(
                pos=node.pos,
                elements=[self._transform_target(e, scope) for e in node.elements],
            )
        return self._transform_node(node, scope)

    def _transform_expr(self, node: IRNode | None, scope: Scope) -> IRNode | None:
        if node is None:
            return None
        if isinstance(node, IRName):
            return self._resolve_name_to_ref(node.name, node.pos, scope, is_write=False)
        if isinstance(node, IRAttribute):
            full = self._collect_attr_chain(node)
            if full and self._lookup(full[0], scope) is not None:
                full_name = ".".join(full)
                sym_id, kind = self._get_or_create_attr_symbol(full_name)
                return self._make_ref(full_name, node.pos, sym_id, kind, is_write=False)

            new_val = self._transform_expr(node.value, scope)
            return IRAttribute(pos=node.pos, value=new_val, attr=node.attr)
        return self._transform_node(node, scope)

    def _resolve_name_to_ref(
        self, name: str, pos: tuple[int, int], scope: Scope, is_write: bool
    ) -> IRNode:
        r = self._lookup(name, scope)
        if r:
            sym_id, kind = r
            return self._make_ref(name, pos, sym_id, kind, is_write)
        return IRName(pos=pos, name=name)

    def _collect_attr_chain(self, node: IRAttribute) -> list[str] | None:
        parts = []
        cur: IRNode = node
        while isinstance(cur, IRAttribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, IRName):
            parts.append(cur.name)
            parts.reverse()
            return parts
        return None

    def _lookup(self, name: str, scope: Scope) -> tuple[int, str] | None:
        s: Scope | None = scope
        while s:
            if name in s.symbols:
                return s.symbols[name]
            s = s.parent
        for ext in self.external_scopes.values():
            if name in ext.symbols:
                return ext.symbols[name]
        return None

    def _get_or_create_attr_symbol(self, full_name: str) -> tuple[int, str]:

        for sym in self.symbol_table.all():
            if sym.name == full_name:
                return sym.id, sym.kind

        for ext in self.external_scopes.values():
            if full_name in ext.symbols:
                return ext.symbols[full_name]

        sym = self.symbol_table.create(full_name, "import")
        return sym.id, sym.kind

    def _make_ref(
        self, name: str, pos: tuple[int, int], sym_id: int, kind: str, is_write: bool
    ) -> IRSymbolRef:
        sym = self.symbol_table.get(sym_id)
        ref = IRSymbolRef(pos=pos, name=name, symbol_id=sym_id, kind=kind)
        if sym:
            sym.uses.append(ref)
            if is_write:
                sym.write_count += 1
            else:
                sym.read_count += 1
        return ref

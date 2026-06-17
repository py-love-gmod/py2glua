from __future__ import annotations

from plg_reader import (
    IRAnnotatedAssign,
    IRAssign,
    IRAttribute,
    IRClassDef,
    IRFile,
    IRFor,
    IRFunctionDef,
    IRName,
    IRNode,
    IRTransformer,
    IRTuple,
    IRWithItem,
)
from utils import Shutdown

from .dt import Scope, Symbol, SymbolRefIR
from .shared import module_name_from_relative_path


class SymlinkResolver:
    """Строит символьную таблицу и заменяет IRName/статические IRAttribute на SymbolRefIR."""

    @classmethod
    def resolve(
        cls, irs: dict[str, IRFile]
    ) -> tuple[dict[str, IRFile], dict[str, Scope], dict[int, Scope]]:
        module_scopes, node_to_scope = cls._collect_definitions(irs)
        irs = cls._replace_names(irs, module_scopes, node_to_scope)
        return irs, module_scopes, node_to_scope

    @classmethod
    def _collect_definitions(
        cls, irs: dict[str, IRFile]
    ) -> tuple[dict[str, Scope], dict[int, Scope]]:
        module_scopes: dict[str, Scope] = {}
        for rel_path in irs:
            mod_name = module_name_from_relative_path(rel_path)
            module_scopes[rel_path] = Scope(
                parent=None, symbols={}, kind="module", module_name=mod_name
            )

        node_to_scope: dict[int, Scope] = {}
        for rel_path, ir_file in irs.items():
            scope = module_scopes[rel_path]
            mod_name = scope.module_name
            scope.symbols[mod_name] = Symbol(
                name=mod_name,
                scope=scope,
                kind="module",
                node=ir_file,
                qualified_name=mod_name,
            )
            cls._collect_module_body(ir_file.body, scope, node_to_scope)

        return module_scopes, node_to_scope

    @staticmethod
    def _collect_module_body(
        body: list[IRNode],
        scope: Scope,
        node_to_scope: dict[int, Scope],
    ) -> None:
        _collect_defs(body, scope, is_class_body=False, node_to_scope=node_to_scope)

    @classmethod
    def _replace_names(
        cls,
        irs: dict[str, IRFile],
        module_scopes: dict[str, Scope],
        node_to_scope: dict[int, Scope],
    ) -> dict[str, IRFile]:
        transformer = _SymlinkTransformer(module_scopes, node_to_scope)
        for rel_path, ir_file in irs.items():
            transformer.current_path = rel_path
            transformer.current_scope = module_scopes[rel_path]
            new_body = []
            for stmt in ir_file.body:
                result = transformer.visit(stmt)
                if result is not None:
                    new_body.append(result)

            ir_file.body = new_body

        return irs


def _collect_defs(
    nodes: list[IRNode],
    current: Scope,
    is_class_body: bool,
    node_to_scope: dict[int, Scope],
) -> None:
    for node in nodes:
        if isinstance(node, IRFunctionDef):
            func_scope = Scope(
                parent=current.parent if is_class_body else current,
                symbols={},
                kind="function",
            )
            node_to_scope[id(node)] = func_scope

            if current.kind == "class" and current.self_symbol:
                qname = f"{current.self_symbol.name}.{node.name}"

            elif current.kind == "module":
                qname = f"{current.module_name}.{node.name}"

            else:
                qname = node.name

            current.symbols[node.name] = Symbol(
                name=node.name,
                scope=current,
                kind="function",
                node=node,
                qualified_name=qname,
            )
            for p in node.params:
                func_scope.symbols[p.name] = Symbol(
                    name=p.name,
                    scope=func_scope,
                    kind="param",
                    node=p,
                    qualified_name=p.name,
                )
            _collect_defs(node.body, func_scope, False, node_to_scope)

        elif isinstance(node, IRClassDef):
            class_scope = Scope(parent=current, symbols={}, kind="class")
            node_to_scope[id(node)] = class_scope

            if current.kind == "module":
                qname = f"{current.module_name}.{node.name}"

            else:
                qname = node.name

            class_sym = Symbol(
                name=node.name,
                scope=current,
                kind="class",
                node=node,
                qualified_name=qname,
            )
            class_scope.self_symbol = class_sym
            current.symbols[node.name] = class_sym
            _collect_defs(node.body, class_scope, True, node_to_scope)

        elif isinstance(node, (IRAssign, IRAnnotatedAssign)):
            targets = node.targets if isinstance(node, IRAssign) else [node.target]
            for t in targets:
                _collect_target(t, current)

        elif isinstance(node, IRFor):
            _collect_target(node.target, current)
            _collect_defs(node.body, current, is_class_body, node_to_scope)

        elif isinstance(node, IRWithItem):
            if node.optional_vars:
                _collect_target(node.optional_vars, current)


def _collect_target(target: IRNode, scope: Scope) -> None:
    if isinstance(target, IRName):
        if scope.kind == "module":
            qname = f"{scope.module_name}.{target.name}"

        elif scope.kind == "class" and scope.self_symbol:
            qname = f"{scope.self_symbol.name}.{target.name}"

        else:
            qname = target.name

        scope.symbols[target.name] = Symbol(
            name=target.name,
            scope=scope,
            kind="variable",
            node=target,
            qualified_name=qname,
        )

    elif isinstance(target, IRTuple):
        for elem in target.elements:
            _collect_target(elem, scope)


class _SymlinkTransformer(IRTransformer):
    def __init__(
        self,
        module_scopes: dict[str, Scope],
        node_to_scope: dict[int, Scope],
    ):
        super().__init__()
        self.module_scopes = module_scopes
        self.node_to_scope = node_to_scope
        self.current_scope: Scope | None = None
        self.scope_stack: list[Scope] = []
        self.global_symbols: dict[str, Symbol] = {}
        self.current_path: str = ""

        self.module_name_to_scope: dict[str, Scope] = {}
        for scope in module_scopes.values():
            if scope.kind == "module" and scope.module_name:
                self.module_name_to_scope[scope.module_name] = scope

    def _error(self, msg: str, pos: tuple[int, int]) -> None:
        location = f"{self.current_path}:{pos[0]}:{pos[1]}"
        raise Shutdown.user_error(f"{msg} в {location}")

    def _softwear_error(self, msg: str, pos: tuple[int, int] | None = None) -> None:
        if pos is not None:
            location = f"{self.current_path}:{pos[0]}:{pos[1]}"
            raise Shutdown.softwear_error(f"{msg} в {location}")

        raise Shutdown.softwear_error(msg)

    def visit(self, node: IRNode | None) -> IRNode | None:
        if node is None:
            return None

        if isinstance(node, (IRFunctionDef, IRClassDef)):
            inner_scope = self.node_to_scope.get(id(node))
            if inner_scope is not None:
                if self.current_scope is not None:
                    self.scope_stack.append(self.current_scope)

                else:
                    self._softwear_error(
                        "current_scope is None при входе в функцию/класс",
                        node.pos,
                    )

                self.current_scope = inner_scope
                result = super().visit(node)
                self.current_scope = self.scope_stack.pop()
                return result

        return super().visit(node)

    def visit_IRName(self, node: IRName) -> IRNode:
        current = self.current_scope
        if current is None:
            self._error(f"Неразрешённое имя '{node.name}'", node.pos)

        scope = current
        while scope is not None:
            sym = scope.symbols.get(node.name)
            if sym is not None:
                ref = SymbolRefIR(pos=node.pos, symbol=sym)
                sym.references.append(ref)
                return ref

            scope = scope.parent

        if node.name not in self.global_symbols:
            kind = "global"
            scope_ref = None
            qname = node.name
            if node.name in self.module_name_to_scope:
                kind = "module"
                scope_ref = self.module_name_to_scope[node.name]
                qname = node.name

            self.global_symbols[node.name] = Symbol(
                name=node.name,
                scope=current,  # pyright: ignore[reportArgumentType]
                kind=kind,
                node=node,
                qualified_name=qname,
                scope_ref=scope_ref,
            )

        global_sym = self.global_symbols[node.name]
        ref = SymbolRefIR(pos=node.pos, symbol=global_sym)
        global_sym.references.append(ref)
        return ref

    def visit_IRAttribute(self, node: IRAttribute) -> IRNode:
        new_value = self.visit(node.value)
        if new_value is None:
            self._error(
                f"Не удалось разрешить левую часть атрибута '{node.attr}'",
                node.pos,
            )

        assert new_value is not None
        node.value = new_value

        if isinstance(new_value, SymbolRefIR):
            base_sym = new_value.symbol
            if base_sym.scope_ref is not None:
                target_scope = base_sym.scope_ref

            elif base_sym.kind == "class":
                target_scope = self.node_to_scope.get(id(base_sym.node))

            else:
                target_scope = None

            if target_scope is not None:
                attr_sym = target_scope.symbols.get(node.attr)
                if attr_sym is not None:
                    ref = SymbolRefIR(pos=node.pos, symbol=attr_sym)
                    attr_sym.references.append(ref)
                    return ref

                else:
                    self._error(
                        f"Модуль/класс '{base_sym.name}' не содержит атрибута '{node.attr}'",
                        node.pos,
                    )

        return node

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
    get_cpus_and_executor,
)
from utils import Shutdown

from .shared import module_name_from_relative_path

ScopeKind = Literal["module", "class", "function", "builtins"]
SymbolKind = Literal["module", "class", "function", "variable", "param", "imported"]


@dataclass
class SymbolRefIR(IRNode):
    symbol: Symbol


@dataclass
class Scope:
    parent: Scope | None
    symbols: dict[str, Symbol]  # локальные имена + экспортируемые (для module/class)
    kind: ScopeKind
    module_name: str = ""  # для module - полное имя, иначе пусто


@dataclass
class Symbol:
    name: str
    scope: Scope  # где определён (не всегда совпадает с местом хранения)
    kind: SymbolKind
    node: IRNode  # узел-источник
    references: list[IRNode] = field(default_factory=list)


class SymlinkResolver:
    """Строит символьную таблицу и заменяет IRName/статические IRAttribute на SymbolRefIR."""

    @classmethod
    def resolve(
        cls, irs: dict[str, IRFile]
    ) -> tuple[dict[str, IRFile], dict[str, Scope], dict[IRNode, Scope]]:
        """
        Возвращает:
            - изменённый irs (с SymbolRefIR)
            - module_scopes (ключ – rel_path)
            - node_to_scope (IRFunctionDef/IRClassDef -> внутренний Scope)
        """
        module_scopes, node_to_scope = cls._collect_definitions(irs)
        irs = cls._replace_names(irs, module_scopes, node_to_scope)
        return irs, module_scopes, node_to_scope

    @classmethod
    def _collect_definitions(
        cls, irs: dict[str, IRFile]
    ) -> tuple[dict[str, Scope], dict[IRNode, Scope]]:
        module_scopes: dict[str, Scope] = {}
        for rel_path in irs:
            mod_name = module_name_from_relative_path(rel_path)
            module_scopes[rel_path] = Scope(
                parent=None, symbols={}, kind="module", module_name=mod_name
            )

        node_to_scope: dict[IRNode, Scope] = {}
        cpus, Executor = get_cpus_and_executor()
        workers = min(cpus, len(irs))

        if workers <= 1:
            for rel_path, ir_file in irs.items():
                scope = module_scopes[rel_path]
                mod_name = scope.module_name
                scope.symbols[mod_name] = Symbol(
                    name=mod_name,
                    scope=scope,
                    kind="module",
                    node=ir_file,
                )
                cls._collect_module_body(ir_file.body, scope, node_to_scope)

        else:
            with Executor(max_workers=workers) as executor:
                futures = {}
                for rel_path, ir_file in irs.items():
                    scope = module_scopes[rel_path]
                    mod_name = scope.module_name
                    scope.symbols[mod_name] = Symbol(
                        name=mod_name,
                        scope=scope,
                        kind="module",
                        node=ir_file,
                    )
                    futures[
                        executor.submit(
                            cls._collect_module_body,
                            ir_file.body,
                            scope,
                            node_to_scope,
                        )
                    ] = rel_path

                for future in futures:
                    future.result()

        return module_scopes, node_to_scope

    @staticmethod
    def _collect_module_body(
        body: list[IRNode],
        scope: Scope,
        node_to_scope: dict[IRNode, Scope],
    ) -> None:
        _collect_defs(body, scope, is_class_body=False, node_to_scope=node_to_scope)

    @classmethod
    def _replace_names(
        cls,
        irs: dict[str, IRFile],
        module_scopes: dict[str, Scope],
        node_to_scope: dict[IRNode, Scope],
    ) -> dict[str, IRFile]:
        cpus, Executor = get_cpus_and_executor()
        workers = min(cpus, len(irs))

        if workers <= 1:
            transformer = _SymlinkTransformer(module_scopes, node_to_scope)
            for rel_path, ir_file in irs.items():
                transformer.current_scope = module_scopes[rel_path]
                new_body = []
                for stmt in ir_file.body:
                    result = transformer.visit(stmt)
                    if result is not None:
                        new_body.append(result)

                ir_file.body = new_body

        else:
            with Executor(max_workers=workers) as executor:
                futures = {}
                for rel_path, ir_file in irs.items():
                    futures[
                        executor.submit(
                            cls._replace_in_module,
                            ir_file,
                            module_scopes,
                            node_to_scope,
                            rel_path,
                        )
                    ] = rel_path

                for future in futures:
                    future.result()

        return irs

    @staticmethod
    def _replace_in_module(
        ir_file: IRFile,
        module_scopes: dict[str, Scope],
        node_to_scope: dict[IRNode, Scope],
        rel_path: str,
    ) -> None:
        transformer = _SymlinkTransformer(module_scopes, node_to_scope)
        transformer.current_scope = module_scopes[rel_path]
        new_body = []
        for stmt in ir_file.body:
            result = transformer.visit(stmt)
            if result is not None:
                new_body.append(result)

        ir_file.body = new_body


def _collect_defs(
    nodes: list[IRNode],
    current: Scope,
    is_class_body: bool,
    node_to_scope: dict[IRNode, Scope],
) -> None:
    for node in nodes:
        if isinstance(node, IRFunctionDef):
            func_scope = Scope(
                parent=current.parent if is_class_body else current,
                symbols={},
                kind="function",
            )
            node_to_scope[node] = func_scope
            current.symbols[node.name] = Symbol(
                name=node.name, scope=current, kind="function", node=node
            )
            for p in node.params:
                func_scope.symbols[p.name] = Symbol(
                    name=p.name, scope=func_scope, kind="param", node=p
                )

            _collect_defs(node.body, func_scope, False, node_to_scope)

        elif isinstance(node, IRClassDef):
            class_scope = Scope(parent=current, symbols={}, kind="class")
            node_to_scope[node] = class_scope
            current.symbols[node.name] = Symbol(
                name=node.name, scope=current, kind="class", node=node
            )
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
        scope.symbols[target.name] = Symbol(
            name=target.name, scope=scope, kind="variable", node=target
        )

    elif isinstance(target, IRTuple):
        for elem in target.elements:
            _collect_target(elem, scope)


class _SymlinkTransformer(IRTransformer):
    def __init__(
        self,
        module_scopes: dict[str, Scope],
        node_to_scope: dict[IRNode, Scope],
    ):
        super().__init__()
        self.module_scopes = module_scopes
        self.node_to_scope = node_to_scope
        self.current_scope: Scope | None = None
        self.scope_stack: list[Scope] = []

    def visit(self, node: IRNode | None) -> IRNode | None:
        if node is None:
            return None

        if isinstance(node, (IRFunctionDef, IRClassDef)):
            inner_scope = self.node_to_scope.get(node)
            if inner_scope is not None:
                if self.current_scope is not None:
                    self.scope_stack.append(self.current_scope)

                else:
                    raise Shutdown.softwear_error(
                        "current_scope is None при входе в функцию/класс"
                    )

                self.current_scope = inner_scope
                result = super().visit(node)
                self.current_scope = self.scope_stack.pop()
                return result

        return super().visit(node)

    def visit_IRName(self, node: IRName) -> IRNode:
        if self.current_scope is None:
            raise Shutdown.user_error(f"Неразрешённое имя '{node.name}' на {node.pos}")

        scope = self.current_scope
        while scope is not None:
            sym = scope.symbols.get(node.name)
            if sym is not None:
                ref = SymbolRefIR(pos=node.pos, symbol=sym)
                sym.references.append(ref)
                return ref

            scope = scope.parent

        raise Shutdown.user_error(f"Неразрешённое имя '{node.name}' на {node.pos}")

    def visit_IRAttribute(self, node: IRAttribute) -> IRNode:
        new_value = self.visit(node.value)
        if new_value is None:
            raise Shutdown.user_error(
                f"Не удалось разрешить левую часть атрибута '{node.attr}' на {node.pos}"
            )

        node.value = new_value

        if isinstance(new_value, SymbolRefIR):
            base_sym = new_value.symbol
            if base_sym.kind in ("module", "class"):
                target_scope = None
                if base_sym.kind == "module":
                    for scope in self.module_scopes.values():
                        if scope.module_name == base_sym.name:
                            target_scope = scope
                            break

                elif base_sym.kind == "class":
                    target_scope = self.node_to_scope.get(base_sym.node)

                if target_scope is not None:
                    attr_sym = target_scope.symbols.get(node.attr)
                    if attr_sym is not None:
                        ref = SymbolRefIR(pos=node.pos, symbol=attr_sym)
                        attr_sym.references.append(ref)
                        return ref

                    else:
                        raise Shutdown.user_error(
                            f"Модуль/класс '{base_sym.name}' не содержит атрибута "
                            f"'{node.attr}' на {node.pos}"
                        )

        return node

from __future__ import annotations

from plg_reader import (
    IRAnnotatedAssign,
    IRAssign,
    IRFile,
    IRFor,
    IRImport,
    IRName,
    IRNode,
    IRTransformer,
)
from utils import Shutdown

from ..shared import (
    build_attr_chain,
    module_name_from_relative_path,
    parse_name_entry,
    resolve_target_module,
)


class _ResolveNamesTransformer(IRTransformer):
    def __init__(self, mapping: dict[str, IRNode]):
        super().__init__()
        self._mapping = mapping
        self._is_target = False

    def _with_target_flag(self, node: IRNode, flag: bool) -> IRNode:
        old = self._is_target
        self._is_target = flag
        result = self.visit(node)
        self._is_target = old
        if result is None:
            raise RuntimeError(f"visit вернул None для {type(node).__name__}")

        return result

    def visit_IRName(self, node: IRName) -> IRNode:
        if self._is_target or node.name not in self._mapping:
            return node

        new_node = self._mapping[node.name]
        new_node.pos = node.pos
        return new_node

    def visit_IRAssign(self, node: IRAssign) -> IRNode:
        node.targets = [self._with_target_flag(t, True) for t in node.targets]
        node.value = self._with_target_flag(node.value, False)
        return node

    def visit_IRAnnotatedAssign(self, node: IRAnnotatedAssign) -> IRNode:
        node.target = self._with_target_flag(node.target, True)
        if node.annotation is not None:
            node.annotation = self._with_target_flag(node.annotation, False)

        if node.value is not None:
            node.value = self._with_target_flag(node.value, False)

        return node

    def visit_IRFor(self, node: IRFor) -> IRNode:
        node.target = self._with_target_flag(node.target, True)
        node.iter = self._with_target_flag(node.iter, False)
        node.body = [self._with_target_flag(stmt, False) for stmt in node.body]
        return node


class ImportResolver:
    """Проверяет допустимость импортов и заменяет имена на атрибутные цепочки."""

    @classmethod
    def resolve_imports(
        cls,
        irs: dict[str, IRFile],
        builtin_modules: frozenset[str] = frozenset(),
    ) -> dict[str, IRFile]:
        if not irs:
            return {}

        allowed = frozenset(module_name_from_relative_path(k) for k in irs)
        for rel_key, ir_file in irs.items():
            current_module = module_name_from_relative_path(rel_key)

            mapping: dict[str, IRNode] = {}
            new_allowed: list[IRImport] = []
            new_builtin: list[IRImport] = []

            for imp in ir_file.imports:
                target = resolve_target_module(imp, current_module)

                if target in builtin_modules:
                    new_builtin.append(imp)

                elif target in allowed:
                    if imp.is_from:
                        for entry in imp.names:
                            orig, alias = parse_name_entry(entry)
                            chain = build_attr_chain(
                                target.split(".") + [orig], imp.pos
                            )
                            mapping[alias] = chain

                    else:
                        for entry in imp.names:
                            orig, alias = parse_name_entry(entry)
                            chain = build_attr_chain(target.split("."), imp.pos)
                            mapping[alias] = chain

                    new_allowed.append(
                        IRImport(
                            pos=imp.pos,
                            modules=[target],
                            names=[],
                            is_from=False,
                            level=0,
                        )
                    )

                else:
                    Shutdown.user_error(
                        f"Импорт внешнего модуля '{target}' запрещён "
                        f"в файле {rel_key} (строка {imp.pos[0]})"
                    )

            ir_file.imports = new_allowed + new_builtin
            if mapping:
                trans = _ResolveNamesTransformer(mapping)
                ir_file.body = [trans._with_target_flag(s, False) for s in ir_file.body]

        return irs

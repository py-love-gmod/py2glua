from __future__ import annotations

from plg_reader import (
    IRCall,
    IRFile,
    IRFunctionDef,
    IRNode,
    IRParam,
    IRTransformer,
)
from utils import Shutdown

from ..dt import CompilationUnit
from ..symbol_table import Scope, SymbolRefIR


class CallNormalizer:
    @classmethod
    def normalize(cls, unit: CompilationUnit) -> CompilationUnit:
        transformer = _CallNormalizerTransformer(unit.node_to_scope)
        for rel_path, ir_file in unit.irs.items():
            transformer.current_file = ir_file
            transformer.current_path = rel_path
            new_body: list[IRNode] = []
            for stmt in ir_file.body:
                result = transformer.visit(stmt)
                if result is not None:
                    new_body.append(result)

            ir_file.body = new_body

        return unit


class _CallNormalizerTransformer(IRTransformer):
    def __init__(self, node_to_scope: dict[int, Scope]):
        super().__init__()
        self.node_to_scope = node_to_scope
        self.current_file: IRFile | None = None
        self.current_path: str = ""

    def _error(self, msg: str, pos: tuple[int, int]) -> None:
        location = f"{self.current_path}:{pos[0]}:{pos[1]}"
        raise Shutdown.user_error(f"{msg} в {location}")

    def visit_IRCall(self, node: IRCall) -> IRNode | None:
        result = self.generic_visit(node)
        if result is None or not isinstance(result, IRCall):
            return result

        node = result

        func = node.func
        if not isinstance(func, SymbolRefIR):
            return node

        sym = func.symbol
        if sym.kind != "function":
            return node

        func_def = sym.node
        if not isinstance(func_def, IRFunctionDef):
            return node

        params = func_def.params
        if not params and not node.kwargs:
            return node

        real_params: list[IRParam] = []
        param_index: dict[str, int] = {}
        kw_only_names: set[str] = set()
        star_idx = None
        slash_idx = None
        for i, p in enumerate(params):
            if p.kind == "slash":
                slash_idx = i
                continue

            if p.kind == "star":
                star_idx = i
                continue

            if p.kind in ("star_arg", "kw_arg"):
                continue

            real_params.append(p)
            idx = len(real_params) - 1
            param_index[p.name] = idx
            if star_idx is not None and i > star_idx:
                kw_only_names.add(p.name)

        new_args: list[IRNode | None] = list(node.args)
        for name in kw_only_names:
            idx = param_index[name]
            if idx < len(new_args) and new_args[idx] is not None:
                self._error(
                    f"Параметр '{name}' нельзя передавать позиционно (keyword-only)",
                    node.pos,
                )

        for kw_name, kw_value in node.kwargs.items():
            if kw_name not in param_index:
                self._error(
                    f"Функция '{func_def.name}' не имеет параметра '{kw_name}'",
                    node.pos,
                )

            idx = param_index[kw_name]
            param = real_params[idx]

            if slash_idx is not None:
                orig_idx = params.index(param)
                if orig_idx < slash_idx:
                    self._error(
                        f"Параметр '{kw_name}' нельзя передавать по имени (позиционный-only)",
                        node.pos,
                    )

            if idx < len(new_args) and new_args[idx] is not None:
                self._error(
                    f"Аргумент '{kw_name}' передан и как позиционный, и как именованный",
                    node.pos,
                )

            while len(new_args) <= idx:
                new_args.append(None)
            new_args[idx] = kw_value

        for idx, param in enumerate(real_params):
            if idx < len(new_args) and new_args[idx] is not None:
                continue

            if param.default is not None:
                while len(new_args) <= idx:
                    new_args.append(None)

                new_args[idx] = param.default

            else:
                self._error(
                    f"Нет значения для параметра '{param.name}' функции '{func_def.name}'",
                    node.pos,
                )

        node.args = [arg for arg in new_args if arg is not None]
        node.kwargs = {}
        return node

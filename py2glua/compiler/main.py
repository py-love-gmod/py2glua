from collections.abc import Callable

from plg_reader import IRFile
from utils import Config

from .analysis import LocalSymbolResolver, Scope, SymbolTable
from .dependency_graph import DependencyGraph
from .lua_dumper import dump_to_lua
from .simplification import AliasResolver


def dump_irs_to_folder(
    irs: dict[str, IRFile],
    symbol_tables: dict[str, SymbolTable] | None = None,
) -> None:
    out = Config.get_path_output()
    out.mkdir(parents=True, exist_ok=True)
    for rel_path, ir_file in irs.items():
        target = out / rel_path
        target = target.with_suffix(".lua")
        target.parent.mkdir(parents=True, exist_ok=True)
        sym = symbol_tables.get(rel_path) if symbol_tables else None
        lua_code = dump_to_lua(ir_file, sym)
        target.write_text(lua_code, encoding="utf-8")


class Compiler:
    _simplification_pass: list[Callable[[IRFile], IRFile]] = [
        AliasResolver.resolve,
    ]

    _graph: DependencyGraph
    _symbol_tables: dict[str, SymbolTable] = {}

    @classmethod
    def _apply_simple_irs_transforms(cls, irs: dict[str, IRFile]) -> None:
        for path, ir in irs.items():
            for pass_ in cls._simplification_pass:
                ir = pass_(ir)
            irs[path] = ir

    @classmethod
    def _check_cycles(cls, irs: dict[str, IRFile]) -> None:
        graph = DependencyGraph.build(irs, Config.get_path_input())
        graph.check_cycles()
        cls._graph = graph

    @classmethod
    def _topological_sort(cls) -> list[str]:
        """Возвращает список файлов в порядке возрастания зависимостей (листья первыми)."""
        in_degree = {
            node: len(cls._graph.reverse.get(node, [])) for node in cls._graph.nodes
        }
        queue = [node for node, deg in in_degree.items() if deg == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in cls._graph.forward.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    @classmethod
    def _build_symbol_tables(cls, irs: dict[str, IRFile]) -> dict[str, SymbolTable]:
        """
        Строит символьные таблицы для всех файлов в правильном порядке
        с учётом внешних зависимостей (external_scopes).
        Заменяет IR каждого файла на трансформированное (с IRSymbolRef).
        Возвращает словарь путь -> SymbolTable.
        """
        ordered_files = cls._topological_sort()
        global_tables: dict[str, SymbolTable] = {}

        for path in ordered_files:
            ir = irs[path]
            external_scopes = {}
            for dep in cls._graph.forward.get(path, []):
                if dep in global_tables:
                    scope = Scope()
                    for sym in global_tables[dep].all():
                        if sym.kind in (
                            "function",
                            "class",
                            "variable",
                            "module",
                            "import",
                        ):
                            scope.symbols[sym.name] = (sym.id, sym.kind)

                    external_scopes[dep] = scope

            new_ir, local_table = LocalSymbolResolver.resolve(ir, external_scopes)
            irs[path] = new_ir
            global_tables[path] = local_table

        return global_tables

    @classmethod
    def build(cls, irs: dict[str, IRFile]) -> None:
        cls._check_cycles(irs)
        cls._apply_simple_irs_transforms(irs)
        cls._symbol_tables = cls._build_symbol_tables(irs)
        dump_irs_to_folder(irs, cls._symbol_tables)

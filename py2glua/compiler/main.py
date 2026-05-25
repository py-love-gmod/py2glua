from collections.abc import Callable

from plg_reader import IRFile
from utils import Config

from .dependency_graph import DependencyGraph


class Compiler:
    _simplification_pass: list[Callable[[IRFile], IRFile]] = []
    _graph: DependencyGraph

    @classmethod
    def _apply_simplification(cls, ir: IRFile) -> IRFile:
        for passs in cls._simplification_pass:
            ir = passs(ir)

        return ir

    @classmethod
    def _check_cycles(cls, irs: dict[str, IRFile]) -> None:
        graph = DependencyGraph.build(irs, Config.get_path_input())
        graph.check_cycles()
        cls._graph = graph  # Базово это сделано для будущей инкрементальной сбокри. Пока что пихуй

    @classmethod
    def _apply_to_irs(cls, irs: dict[str, IRFile], func: Callable[[IRFile], IRFile]):
        for path, ir in irs.items():
            irs[path] = func(ir)

    @classmethod
    def build(cls, irs: dict[str, IRFile]) -> None:
        cls._check_cycles(irs)

        cls._apply_to_irs(irs, cls._apply_simplification)

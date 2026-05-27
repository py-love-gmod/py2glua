from collections.abc import Callable

from plg_reader import IRFile
from utils import Config

from .dependency_graph import DependencyGraph
from .simplification import AliasResolver


def _dump_irs(irs: dict[str, IRFile]) -> None:
    for path, ir in irs.items():
        print(f"{path=}")
        print(ir.pretty())
        print("-" * 20)


class Compiler:
    _simplification_pass: list[Callable[[IRFile], IRFile]] = [
        AliasResolver.resolve,
    ]

    _graph: DependencyGraph

    @classmethod
    def _apply_simple_irs_transforms(
        cls,
        irs: dict[str, IRFile],
        *list_passes: list[Callable[[IRFile], IRFile]],
    ) -> None:
        for path, ir in irs.items():
            for pass_type in list_passes:
                for pass_ in pass_type:
                    ir = pass_(ir)

            irs[path] = ir

    @classmethod
    def _check_cycles(cls, irs: dict[str, IRFile]) -> None:
        graph = DependencyGraph.build(irs, Config.get_path_input())
        graph.check_cycles()
        cls._graph = graph  # Базово это сделано для будущей инкрементальной сбокри. Пока что пихуй

    @classmethod
    def build(cls, irs: dict[str, IRFile]) -> None:
        cls._check_cycles(irs)

        cls._apply_simple_irs_transforms(
            irs,
            cls._simplification_pass,
        )

        _dump_irs(irs)

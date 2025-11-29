from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from ..py.ir_builder import PyIRBuilder
from ..py.ir_dataclass import PyIRFile


class Compiler:
    def __init__(
        self,
        version: str,
        project_root: Path,
        config: dict,
        file_passes: Sequence,
        project_passes: Sequence,
    ):
        self.version: str = version
        self.project_root = project_root.resolve()

        self.modules: Dict[str, PyIRFile] = {}

        self.path_to_name: Dict[Path, str] = {}

        self.config = config

        self.file_passes = list(file_passes)
        self.project_passes = list(project_passes)

    def build(self, entry_points: Sequence[Path]) -> List[PyIRFile]:
        for ep in entry_points:
            self.load_file(ep)

        project = list(self.modules.values())
        project = self._run_project_passes(project)

        return project

    def load_file(self, path: Path) -> PyIRFile:
        path = path.resolve()

        module_name = self._module_name_from_path(path)
        if module_name in self.modules:
            return self.modules[module_name]

        text = path.read_text("utf8")
        ir = PyIRBuilder.build_file(text, path)

        ir.context.meta["module"] = module_name
        self.modules[module_name] = ir
        self.path_to_name[path] = module_name

        self._run_file_passes(ir)

        return ir

    def ensure_loaded(self, module_name: str) -> None:
        if module_name in self.modules:
            return

        path = self._module_to_file(module_name)
        if path is None:
            raise FileNotFoundError(f"Module '{module_name}' not found")

        self.load_file(path)

    def _run_file_passes(self, ir: PyIRFile):
        for p in self.file_passes:
            p.run(ir, self)

    def _run_project_passes(self, project: List[PyIRFile]) -> List[PyIRFile]:
        for p in self.project_passes:
            result = p.run(project, self)
            if result is not None:
                project = result

        return project

    def _module_name_from_path(self, path: Path) -> str:
        relative = path.relative_to(self.project_root)
        parts = relative.with_suffix("").parts
        return ".".join(parts)

    def _module_to_file(self, module_name: str) -> Path | None:
        p = module_name.split(".")
        candidate = self.project_root.joinpath(*p).with_suffix(".py")
        return candidate if candidate.exists() else None

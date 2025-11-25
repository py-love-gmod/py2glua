from __future__ import annotations

from typing import List

from ..py.py_ir_dataclass import PyIRFile


class Compiler:
    def __init__(self):
        self.file_passes = []
        self.project_passes = []

    def add_file_pass(self, p):
        if isinstance(p, list):
            self.file_passes.extend(p)
        else:
            self.file_passes.append(p)

    def add_project_pass(self, p):
        if isinstance(p, list):
            self.project_passes.extend(p)
        else:
            self.project_passes.append(p)

    def run_file_passes(self, file: PyIRFile) -> PyIRFile:
        for p in self.file_passes:
            p.run(file)

        return file

    def run_project_passes(self, project: List[PyIRFile]) -> List[PyIRFile]:
        for p in self.project_passes:
            p.run(project)

        return project

from pathlib import Path


class TypeRegistry:
    def __init__(self):
        self._type_to_module: dict[str, Path] = {}

    def register_type(self, type_name: str, file_path: Path) -> None:
        self._type_to_module[type_name] = file_path

    def get_import_for_type(self, type_name: str, current_file: Path) -> str | None:
        if type_name in self._type_to_module:
            module_path = self._type_to_module[type_name]
            if module_path.resolve() == current_file.resolve():
                return None

            stem = module_path.stem
            return f"from .{stem} import {type_name}"

        return None

    def all_types(self) -> list[str]:
        return sorted(self._type_to_module.keys())

    def get_module_for_type(self, type_name: str) -> str | None:
        if type_name in self._type_to_module:
            return self._type_to_module[type_name].stem

        return None

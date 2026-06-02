from pathlib import Path
from typing import Optional, Set


class TypeRegistry:
    _type_to_module: dict[str, str] = {}
    _type_to_file: dict[str, Path] = {}
    _module_to_types: dict[Path, Set[str]] = {}
    _file_to_types: dict[Path, Set[str]] = {}

    @classmethod
    def register(cls, file_path: Path, types: Set[str]) -> None:
        cls._file_to_types[file_path] = types
        module_name = file_path.stem
        for t in types:
            cls._type_to_module[t] = module_name
            cls._type_to_file[t] = file_path

    @classmethod
    def get_module_for_type(cls, type_name: str) -> Optional[str]:
        return cls._type_to_module.get(type_name)

    @classmethod
    def get_file_for_type(cls, type_name: str) -> Optional[Path]:
        return cls._type_to_file.get(type_name)

    @classmethod
    def get_all_types(cls) -> Set[str]:
        return set(cls._type_to_module.keys())

    @classmethod
    def get_import_for_type(cls, type_name: str, current_file: Path) -> Optional[str]:
        target_file = cls.get_file_for_type(type_name)
        if target_file and target_file != current_file:
            module_name = cls.get_module_for_type(type_name)
            return f"from .{module_name} import {type_name}"

        return None

    @classmethod
    def clear(cls) -> None:
        cls._type_to_module.clear()
        cls._type_to_file.clear()
        cls._file_to_types.clear()

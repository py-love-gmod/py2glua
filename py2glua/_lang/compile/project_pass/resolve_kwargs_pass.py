from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

from ...py.py_ir_dataclass import (
    PyIRCall,
    PyIRFile,
    PyIRFunctionDef,
)
from ..compile_exception import CompileError
from ..compiler import Compiler


class ResolveKwargsProjectPass:
    @classmethod
    def run(cls, project: List[PyIRFile], compiler: Compiler) -> List[PyIRFile]:
        module_by_file: Dict[PyIRFile, str | None] = {}
        for f in project:
            module_by_file[f] = cls._get_module_name(f)

        func_sigs: Dict[str, List[str]] = {}
        for f in project:
            mod_name = module_by_file.get(f)
            if not mod_name:
                continue

            for node in f.body:
                if isinstance(node, PyIRFunctionDef):
                    full_name = f"{mod_name}.{node.name}"
                    param_names = list(node.signature.keys())
                    func_sigs[full_name] = param_names

        visible_per_file: Dict[PyIRFile, Dict[str, str]] = {}
        for f in project:
            visible: Dict[str, str] = {}

            mod_name = module_by_file.get(f)
            if mod_name:
                for node in f.body:
                    if isinstance(node, PyIRFunctionDef):
                        full_name = f"{mod_name}.{node.name}"
                        visible[node.name] = full_name

            ctx = getattr(f, "context", None)
            if ctx is not None:
                imports = ctx.meta.get("imports", [])
                aliases = ctx.meta.get("aliases", {})

                for mod in imports:
                    if not isinstance(mod, str):
                        continue

                    parts = mod.split(".")
                    if not parts:
                        continue

                    simple = parts[-1]
                    visible.setdefault(simple, mod)

                for alias_name, real_name in aliases.items():
                    if isinstance(alias_name, str) and isinstance(real_name, str):
                        visible[alias_name] = real_name

            visible_per_file[f] = visible

        for f in project:
            visible = visible_per_file.get(f, {})
            for node in f.walk():
                if not isinstance(node, PyIRCall):
                    continue

                if not node.args_kw:
                    continue

                if not node.name:
                    raise CompileError(
                        "Keyword arguments are only supported for direct function calls by name.\n"
                        f"FILE: {f.path}\n"
                        f"LINE|OFFSET: {node.line}|{node.offset}"
                    )

                func_name = node.name

                full_symbol = cls._resolve_symbol_for_call(
                    file=f,
                    func_name=func_name,
                    module_by_file=module_by_file,
                    visible=visible,
                )

                param_names = func_sigs.get(full_symbol)
                if param_names is None:
                    raise CompileError(
                        f"Keyword arguments are used in call to '{func_name}', "
                        f"but function signature is unknown.\n"
                        f"FILE: {f.path}\n"
                        f"LINE|OFFSET: {node.line}|{node.offset}"
                    )

                cls._rewrite_call_kwargs(node, param_names, f)

        return project

    # region Helpers
    @staticmethod
    def _get_module_name(f: PyIRFile) -> str | None:
        ctx = getattr(f, "context", None)
        if ctx is not None:
            mod = ctx.meta.get("module")
            if isinstance(mod, str) and mod:
                return mod

        if f.path is None:
            return None

        p: Path = f.path.with_suffix("")
        if not p.parts:
            return None

        return ".".join(p.parts)

    @staticmethod
    def _resolve_symbol_for_call(
        file: PyIRFile,
        func_name: str,
        module_by_file: Dict[PyIRFile, str | None],
        visible: Dict[str, str],
    ) -> str:
        if func_name in visible:
            return visible[func_name]

        mod_name = module_by_file.get(file)
        if mod_name:
            return f"{mod_name}.{func_name}"

        return func_name

    @staticmethod
    def _rewrite_call_kwargs(
        call: PyIRCall,
        param_names: List[str],
        file: PyIRFile,
    ) -> None:
        pos_args = list(call.args_p)
        kw_args = call.args_kw or {}

        n_params = len(param_names)

        if len(pos_args) > n_params:
            raise CompileError(
                f"Too many positional arguments for function '{call.name}'.\n"
                f"FILE: {file.path}\n"
                f"LINE|OFFSET: {call.line}|{call.offset}"
            )

        idx_map: Dict[str, int] = {}
        for kw_name in kw_args.keys():
            if kw_name not in param_names:
                raise CompileError(
                    f"Unknown keyword argument '{kw_name}' for function '{call.name}'.\n"
                    f"FILE: {file.path}\n"
                    f"LINE|OFFSET: {call.line}|{call.offset}"
                )

            idx = param_names.index(kw_name)
            idx_map[kw_name] = idx

        for kw_name, idx in idx_map.items():
            if idx < len(pos_args):
                raise CompileError(
                    f"Multiple values for argument '{kw_name}' in call to '{call.name}'.\n"
                    f"FILE: {file.path}\n"
                    f"LINE|OFFSET: {call.line}|{call.offset}"
                )

        if idx_map:
            sorted_idx = sorted(idx_map.values())
            expected_start = len(pos_args)
            expected = list(range(expected_start, expected_start + len(sorted_idx)))

            if sorted_idx != expected:
                raise CompileError(
                    "Non-contiguous keyword arguments are not supported yet. "
                    "All keyword arguments must form a contiguous tail after positional arguments.\n"
                    f"FUNCTION: {call.name}\n"
                    f"FILE: {file.path}\n"
                    f"LINE|OFFSET: {call.line}|{call.offset}"
                )

        new_args = pos_args[:]

        used_indices: Set[int] = set()
        for idx in sorted(idx_map.values()):
            pname = param_names[idx]
            new_args.append(kw_args[pname])
            used_indices.add(idx)

        call.args_p = new_args
        call.args_kw = {}

    # endregion

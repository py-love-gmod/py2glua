from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from ....._config import Py2GluaConfig
from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRImport,
    PyIRImportType,
    PyIRNode,
    PyIRVarUse,
)


class NormalizeImportsPass:
    """
    Обрабатывает:
      - from-import импорты

    Назначение:
      - нормализует импорты к форме "import module"
      - выполняет символическую замену импортированных имён:
          y  -> module.y
          z  -> module.y   (для "from module import y as z")
    """

    _reexport_cache: dict[
        tuple[int, tuple[str, ...]],
        dict[str, tuple[tuple[str, ...], str]],
    ] = {}

    @staticmethod
    def _iter_local_base_paths(mod_parts: tuple[str, ...]) -> Iterable[Path]:
        src_root = Py2GluaConfig.source.resolve()
        yield src_root.joinpath(*mod_parts)

        if mod_parts and src_root.name == mod_parts[0]:
            yield src_root.joinpath(*mod_parts[1:])

    @classmethod
    def _local_mod_file_exists(cls, mod_parts: tuple[str, ...]) -> bool:
        for base in cls._iter_local_base_paths(mod_parts):
            if base.with_suffix(".py").exists():
                return True

        return False

    @classmethod
    def _local_pkg_init_exists(cls, mod_parts: tuple[str, ...]) -> bool:
        for base in cls._iter_local_base_paths(mod_parts):
            if base.is_dir() and (base / "__init__.py").exists():
                return True

        return False

    @classmethod
    def _local_module_exists(cls, mod_parts: tuple[str, ...]) -> bool:
        for base in cls._iter_local_base_paths(mod_parts):
            if base.with_suffix(".py").exists():
                return True
            if base.is_dir():
                return True
        return False

    @staticmethod
    def _internal_mod_file_exists(mod_parts: tuple[str, ...]) -> bool:
        if len(mod_parts) < 2:
            return False

        parent_pkg = ".".join(mod_parts[:-1])
        leaf = mod_parts[-1]
        try:
            root = resources.files(parent_pkg)
            return root.joinpath(f"{leaf}.py").is_file()

        except Exception:
            return False

    @staticmethod
    def _internal_pkg_init_exists(mod_parts: tuple[str, ...]) -> bool:
        pkg_name = ".".join(mod_parts)
        try:
            root = resources.files(pkg_name)
            return root.joinpath("__init__.py").is_file()

        except Exception:
            return False

    @classmethod
    def _internal_module_exists(cls, mod_parts: tuple[str, ...]) -> bool:
        if cls._internal_mod_file_exists(mod_parts):
            return True

        pkg_name = ".".join(mod_parts)
        try:
            resources.files(pkg_name)
            return True
        except Exception:
            return False

    @classmethod
    def _should_try_reexports(
        cls,
        abs_modules: tuple[str, ...],
        itype: PyIRImportType,
    ) -> PyIRImportType | None:
        if itype == PyIRImportType.LOCAL:
            if cls._local_mod_file_exists(abs_modules):
                return None

            return (
                PyIRImportType.LOCAL
                if cls._local_pkg_init_exists(abs_modules)
                else None
            )

        if itype == PyIRImportType.INTERNAL:
            if cls._internal_mod_file_exists(abs_modules):
                return None

            return (
                PyIRImportType.INTERNAL
                if cls._internal_pkg_init_exists(abs_modules)
                else None
            )

        if itype == PyIRImportType.UNKNOWN:
            if cls._local_mod_file_exists(abs_modules):
                return None

            if cls._local_pkg_init_exists(abs_modules):
                return PyIRImportType.LOCAL

            if cls._internal_mod_file_exists(abs_modules):
                return None

            if cls._internal_pkg_init_exists(abs_modules):
                return PyIRImportType.INTERNAL

        return None

    @classmethod
    def run(cls, ir: PyIRFile) -> PyIRFile:
        mapping: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

        new_body: list[PyIRNode] = []
        added_imports: set[tuple[int, tuple[str, ...], int]] = set()

        for node in ir.body:
            if not isinstance(node, PyIRImport):
                new_body.append(node)
                continue

            if not node.if_from:
                new_body.append(node)
                continue

            abs_modules = cls._absolutize_from_module(ir, node)
            abs_level = 0

            for n in node.names:
                orig, alias = cls._orig_and_alias(n)
                local_name = alias or orig

                resolved_modules, resolved_attr_parts = cls._resolve_export_if_needed(
                    abs_modules=abs_modules,
                    imported_name=orig,
                    itype=node.itype,
                )

                mapping[local_name] = (resolved_modules, resolved_attr_parts)

                k = (int(node.itype), resolved_modules, abs_level)
                if k not in added_imports and not cls._has_plain_import_parts(
                    new_body,
                    itype=node.itype,
                    modules=resolved_modules,
                    level=abs_level,
                ):
                    new_body.append(
                        PyIRImport(
                            line=node.line,
                            offset=node.offset,
                            modules=list(resolved_modules),
                            names=[],
                            if_from=False,
                            level=abs_level,
                            itype=node.itype,
                        )
                    )
                    added_imports.add(k)

            continue

        ir.body = new_body

        if mapping:
            ir.body = cls._rewrite_block(ir.body, mapping)

        return ir

    @classmethod
    def _rewrite_block(
        cls,
        body: list[PyIRNode],
        mapping: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> list[PyIRNode]:
        return [cls._rewrite_node(n, mapping) for n in body]

    @classmethod
    def _rewrite_node(
        cls,
        node: PyIRNode,
        mapping: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> PyIRNode:
        if isinstance(node, PyIRVarUse):
            hit = mapping.get(node.name)
            if hit is None:
                return node

            module_parts, attr_parts = hit
            return cls._build_attr_chain(module_parts, attr_parts, node)

        if not is_dataclass(node):
            return node

        for f in fields(node):
            val = getattr(node, f.name)

            new_val = cls._rewrite_value(val, mapping)
            if new_val is not val:
                setattr(node, f.name, new_val)

        return node

    @classmethod
    def _rewrite_value(
        cls,
        val: Any,
        mapping: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> Any:
        if isinstance(val, PyIRNode):
            return cls._rewrite_node(val, mapping)

        if isinstance(val, list):
            changed = False
            out_list: list[Any] = []
            for x in val:
                nx = cls._rewrite_value(x, mapping)
                changed = changed or (nx is not x)
                out_list.append(nx)

            return out_list if changed else val

        if isinstance(val, dict):
            changed = False
            out_1: dict[Any, Any] = {}
            for k, v in val.items():
                nk = cls._rewrite_value(k, mapping)
                nv = cls._rewrite_value(v, mapping)
                changed = changed or (nk is not k) or (nv is not v)
                out_1[nk] = nv

            return out_1 if changed else val

        if isinstance(val, tuple):
            changed = False
            out_tuple_items: list[Any] = []
            for x in val:
                nx = cls._rewrite_value(x, mapping)
                changed = changed or (nx is not x)
                out_tuple_items.append(nx)
            return tuple(out_tuple_items) if changed else val

        return val

    @classmethod
    def _resolve_export_if_needed(
        cls,
        abs_modules: tuple[str, ...],
        imported_name: str,
        itype: PyIRImportType,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        imported_parts = tuple(imported_name.split("."))
        head = imported_parts[0]
        tail = imported_parts[1:]

        if itype in (
            PyIRImportType.LOCAL,
            PyIRImportType.UNKNOWN,
        ) and cls._local_mod_file_exists(abs_modules):
            return abs_modules, imported_parts

        if itype in (
            PyIRImportType.INTERNAL,
            PyIRImportType.UNKNOWN,
        ) and cls._internal_mod_file_exists(abs_modules):
            return abs_modules, imported_parts

        # (1) reexports (pkg/__init__.py)
        reexp_itype = cls._should_try_reexports(abs_modules, itype)
        if reexp_itype is not None:
            exp = cls._read_reexports_for_package(abs_modules, itype=reexp_itype)
            hit = exp.get(head)
            if hit is not None:
                target_mod, target_attr = hit
                return target_mod, (target_attr, *tail)

        # (2) submodule import: from pkg import submodule
        candidate_mod = (*abs_modules, head)

        if itype in (PyIRImportType.LOCAL, PyIRImportType.UNKNOWN):
            if cls._local_module_exists(candidate_mod):
                return candidate_mod, tail

        if itype in (PyIRImportType.INTERNAL, PyIRImportType.UNKNOWN):
            if cls._internal_module_exists(candidate_mod):
                return candidate_mod, tail

        # (3) обычный атрибут
        return abs_modules, imported_parts

    @classmethod
    def _read_reexports_for_package(
        cls,
        pkg_modules: tuple[str, ...],
        *,
        itype: PyIRImportType,
    ) -> dict[str, tuple[tuple[str, ...], str]]:
        key = (int(itype), pkg_modules)
        cached = cls._reexport_cache.get(key)
        if cached is not None:
            return cached

        out: dict[str, tuple[tuple[str, ...], str]] = {}

        init_py: Path | None = None
        text: str | None = None

        if itype == PyIRImportType.INTERNAL:
            if cls._internal_mod_file_exists(
                pkg_modules
            ) or not cls._internal_pkg_init_exists(pkg_modules):
                cls._reexport_cache[key] = out
                return out

            pkg_name = ".".join(pkg_modules)
            try:
                tr = resources.files(pkg_name)
                with resources.as_file(tr) as pkg_root:
                    p = Path(pkg_root) / "__init__.py"
                    if p.exists():
                        init_py = p
                        text = p.read_text(encoding="utf-8")
            except Exception:
                init_py = None
                text = None

        elif itype == PyIRImportType.LOCAL:
            if cls._local_mod_file_exists(
                pkg_modules
            ) or not cls._local_pkg_init_exists(pkg_modules):
                cls._reexport_cache[key] = out
                return out

            for base in cls._iter_local_base_paths(pkg_modules):
                p = base / "__init__.py"
                if p.exists():
                    try:
                        init_py = p
                        text = p.read_text(encoding="utf-8")
                        break

                    except Exception:
                        init_py = None
                        text = None

        if init_py is None or text is None:
            cls._reexport_cache[key] = out
            return out

        try:
            tree = ast.parse(text, filename=str(init_py))
        except SyntaxError:
            cls._reexport_cache[key] = out
            return out

        base_pkg = pkg_modules

        for stmt in tree.body:
            if not isinstance(stmt, ast.ImportFrom):
                continue

            target_mod = cls._absolutize_importfrom_module(
                base_pkg=base_pkg,
                module=stmt.module,
                level=stmt.level or 0,
            )
            if target_mod is None:
                continue

            for a in stmt.names:
                src_name = a.name
                exported_name = a.asname or a.name
                out[exported_name] = (target_mod, src_name)

        cls._reexport_cache[key] = out
        return out

    @staticmethod
    def _absolutize_importfrom_module(
        base_pkg: tuple[str, ...],
        module: str | None,
        level: int,
    ) -> tuple[str, ...] | None:
        if module is None:
            return None

        mod_parts = tuple(module.split(".")) if module else ()

        if level <= 0:
            return mod_parts

        up = level - 1
        if up > len(base_pkg):
            return None

        base = base_pkg[: len(base_pkg) - up]
        return (*base, *mod_parts)

    @classmethod
    def _absolutize_from_module(
        cls,
        ir: PyIRFile,
        node: PyIRImport,
    ) -> tuple[str, ...]:
        mods = tuple(node.modules or ())
        level = int(node.level or 0)
        if level <= 0:
            return mods

        if ir.path is None:
            return mods

        cur_mod = cls._module_parts_from_path(ir.path)
        if cur_mod is None:
            return mods

        cur_pkg = cur_mod
        if cur_pkg and cur_pkg[-1] == "__init__":
            cur_pkg = cur_pkg[:-1]
        else:
            cur_pkg = cur_pkg[:-1]

        up = level - 1
        if up > len(cur_pkg):
            return mods

        base = cur_pkg[: len(cur_pkg) - up]
        return (*base, *mods)

    @staticmethod
    def _module_parts_from_path(path: Path) -> tuple[str, ...] | None:
        p = path.resolve()

        parts = p.parts
        try:
            src_root = Py2GluaConfig.source.resolve()
            rel = p.relative_to(src_root)
        except Exception:
            rel = None

        if rel is None:
            try:
                i = parts.index("py2glua")
                tail = list(parts[i:])

            except ValueError:
                return None

        else:
            tail = list(rel.parts)

        if not tail:
            return None

        last = tail[-1]
        if last.endswith(".py"):
            name = last[:-3]
            tail[-1] = "__init__" if name == "__init__" else name

        return tuple(tail)

    @staticmethod
    def _orig_and_alias(name: str | tuple[str, str]) -> tuple[str, str | None]:
        if isinstance(name, tuple):
            return name[0], name[1]

        return name, None

    @staticmethod
    def _has_plain_import_parts(
        body: list[PyIRNode],
        itype: PyIRImportType,
        modules: tuple[str, ...],
        level: int,
    ) -> bool:
        for n in body:
            if not isinstance(n, PyIRImport):
                continue

            if n.if_from:
                continue

            if n.itype != itype:
                continue

            if tuple(n.modules or ()) == modules and int(n.level or 0) == int(level):
                return True

        return False

    @staticmethod
    def _build_attr_chain(
        module_parts: tuple[str, ...],
        attr_parts: tuple[str, ...],
        src: PyIRVarUse,
    ) -> PyIRNode:
        if not module_parts:
            return src

        expr: PyIRNode = PyIRVarUse(
            line=src.line,
            offset=src.offset,
            name=module_parts[0],
        )

        for p in module_parts[1:]:
            expr = PyIRAttribute(
                line=src.line,
                offset=src.offset,
                value=expr,
                attr=p,
            )

        for a in attr_parts:
            expr = PyIRAttribute(
                line=src.line,
                offset=src.offset,
                value=expr,
                attr=a,
            )

        return expr

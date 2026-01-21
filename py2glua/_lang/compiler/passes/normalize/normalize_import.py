from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any

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
        tuple[str, ...],
        dict[str, tuple[tuple[str, ...], str]],
    ] = {}

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
            out: list[Any] = []
            for x in val:
                nx = cls._rewrite_value(x, mapping)
                changed = changed or (nx is not x)
                out.append(nx)

            return out if changed else val

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
            out: list[Any] = []
            for x in val:
                nx = cls._rewrite_value(x, mapping)
                changed = changed or (nx is not x)
                out.append(nx)
            return tuple(out) if changed else val

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

        if itype == PyIRImportType.INTERNAL:
            exp = cls._read_reexports_for_package(abs_modules)
            hit = exp.get(head)
            if hit is not None:
                target_mod, target_attr = hit
                return target_mod, (target_attr, *tail)

        return abs_modules, imported_parts

    @classmethod
    def _read_reexports_for_package(
        cls,
        pkg_modules: tuple[str, ...],
    ) -> dict[str, tuple[tuple[str, ...], str]]:
        cached = cls._reexport_cache.get(pkg_modules)
        if cached is not None:
            return cached

        out: dict[str, tuple[tuple[str, ...], str]] = {}

        pkg_name = ".".join(pkg_modules)

        try:
            tr = resources.files(pkg_name)

        except Exception:
            cls._reexport_cache[pkg_modules] = out
            return out

        try:
            with resources.as_file(tr) as pkg_root:
                init_py = Path(pkg_root) / "__init__.py"
                if not init_py.exists():
                    cls._reexport_cache[pkg_modules] = out
                    return out

                text = init_py.read_text(encoding="utf-8")

        except Exception:
            cls._reexport_cache[pkg_modules] = out
            return out

        try:
            tree = ast.parse(text, filename=str(init_py))

        except SyntaxError:
            cls._reexport_cache[pkg_modules] = out
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

        cls._reexport_cache[pkg_modules] = out
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

        cur_pkg = cur_mod if cur_mod and cur_mod[-1] != "__init__" else cur_mod
        if cur_pkg and cur_pkg[-1] != "__init__":
            cur_pkg = cur_pkg[:-1]

        else:
            cur_pkg = tuple(p for p in cur_pkg if p != "__init__")

        up = level - 1
        if up > len(cur_pkg):
            return mods

        base = cur_pkg[: len(cur_pkg) - up]
        return (*base, *mods)

    @staticmethod
    def _module_parts_from_path(path: Path) -> tuple[str, ...] | None:
        parts = path.parts
        try:
            i = parts.index("py2glua")

        except ValueError:
            return None

        tail = list(parts[i:])

        if not tail:
            return None

        last = tail[-1]
        if last.endswith(".py"):
            name = last[:-3]
            if name == "__init__":
                tail = tail[:-1]

            else:
                tail[-1] = name

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
            line=src.line, offset=src.offset, name=module_parts[0]
        )

        for p in module_parts[1:]:
            expr = PyIRAttribute(line=src.line, offset=src.offset, value=expr, attr=p)

        for a in attr_parts:
            expr = PyIRAttribute(line=src.line, offset=src.offset, value=expr, attr=a)

        return expr

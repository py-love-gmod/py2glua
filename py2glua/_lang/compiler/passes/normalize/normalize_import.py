from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRImport,
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

    Правила:
      - создаёт/переиспользует import X (не-from)
      - удаляет исходный from-import
    """

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

            modules = tuple(node.modules or ())
            level = int(node.level or 0)

            k = (int(node.itype), modules, level)
            if k not in added_imports and not cls._has_plain_import(new_body, node):
                new_body.append(
                    PyIRImport(
                        line=node.line,
                        offset=node.offset,
                        modules=list(modules),
                        names=[],
                        if_from=False,
                        level=node.level,
                        itype=node.itype,
                    )
                )
                added_imports.add(k)

            for n in node.names or []:
                orig, alias = cls._orig_and_alias(n)
                local_name = alias or orig
                mapping[local_name] = (modules, tuple(orig.split(".")))

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
        out: list[PyIRNode] = []
        for n in body:
            out.append(cls._rewrite_node(n, mapping))

        return out

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
            new_list: list[Any] = []
            for x in val:
                nx = cls._rewrite_value(x, mapping)
                changed = changed or (nx is not x)
                new_list.append(nx)

            return new_list if changed else val

        if isinstance(val, dict):
            changed = False
            new_dict: dict[Any, Any] = {}
            for k, v in val.items():
                nk = cls._rewrite_value(k, mapping)
                nv = cls._rewrite_value(v, mapping)
                changed = changed or (nk is not k) or (nv is not v)
                new_dict[nk] = nv

            return new_dict if changed else val

        if isinstance(val, tuple):
            changed = False
            new_items: list[Any] = []
            for x in val:
                nx = cls._rewrite_value(x, mapping)
                changed = changed or (nx is not x)
                new_items.append(nx)

            return tuple(new_items) if changed else val

        return val

    @staticmethod
    def _orig_and_alias(name: str | tuple[str, str]) -> tuple[str, str | None]:
        if isinstance(name, tuple):
            return name[0], name[1]

        return name, None

    @staticmethod
    def _name_of(name: str | tuple[str, str]) -> str:
        return name[0] if isinstance(name, tuple) else name

    @staticmethod
    def _has_plain_import(body: list[PyIRNode], src: PyIRImport) -> bool:
        for n in body:
            if not isinstance(n, PyIRImport):
                continue

            if n.if_from:
                continue

            if n.itype != src.itype:
                continue

            if (n.modules or []) == (src.modules or []) and int(n.level or 0) == int(
                src.level or 0
            ):
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

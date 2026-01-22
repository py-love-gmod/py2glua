from __future__ import annotations

from typing import Optional

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRClassDef,
    PyIRConstant,
    PyIRDict,
    PyIRDictItem,
    PyIRTuple,
    PyIRVarUse,
)
from ...ir_compiler import PyIRSymbolRef
from .ctx import (
    AnalysisContext,
    AnalysisPass,
    EnumInfo,
    EnumValue,
)


def _get_gmod_special_enum_decorator(node: PyIRClassDef):
    for d in node.decorators:
        if d.name == "CompilerDirective.gmod_special_enum":
            return d

    return None


def _get_class_symbol_id(
    node: PyIRClassDef, ir: PyIRFile, ctx: AnalysisContext
) -> Optional[int]:
    if ir.path is None:
        return None

    table = ctx.file_simbol_data.get(ir.path)
    if table is None:
        return None

    for sid, info in ctx.symbols.items():
        if info.file != ir.path:
            continue

        sym = info.symbol
        if sym.scope != "module":
            continue

        if sym.kind != "class":
            continue

        if sym.name != node.name:
            continue

        return sid

    return None


def _get_class_fqname_by_id(sid: int, ctx: AnalysisContext) -> str:
    info = ctx.symbols.get(sid)
    if info is None:
        return f"<sym:{sid}>"

    return info.fqname


def _is_python_enum_base(base, ctx: AnalysisContext) -> bool:
    if not isinstance(base, PyIRAttribute):
        return False

    if isinstance(base.value, PyIRVarUse):
        return base.value.name == "enum" and base.attr == "Enum"

    if isinstance(base.value, PyIRSymbolRef):
        info = ctx.symbols.get(base.value.sym_id)
        if info is None:
            return False

        fq = info.fqname
        if fq.endswith(".enum") or fq == "enum":
            return base.attr == "Enum"

    return False


class CollectEnumsPass(AnalysisPass):
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        for node in ir.body:
            if not isinstance(node, PyIRClassDef):
                continue

            cls._try_collect_enum(node, ir, ctx)

    @classmethod
    def _try_collect_enum(
        cls, node: PyIRClassDef, ir: PyIRFile, ctx: AnalysisContext
    ) -> None:
        sid = _get_class_symbol_id(node, ir, ctx)
        if sid is None:
            return

        fqname = _get_class_fqname_by_id(sid, ctx)

        deco = _get_gmod_special_enum_decorator(node)
        if deco is not None:
            ctx.enums[sid] = cls._collect_gmod_enum(deco, sid, fqname)
            return

        if any(_is_python_enum_base(b, ctx) for b in node.bases):
            ctx.enums[sid] = cls._collect_python_enum(node, sid, fqname)
            return

    @staticmethod
    def _collect_gmod_enum(deco, sid: int, fqname: str) -> EnumInfo:
        fields_node = deco.args_kw.get("fields")
        if fields_node is None:
            raise RuntimeError(f"gmod_special_enum on {fqname} requires fields=")

        if not isinstance(fields_node, PyIRDict):
            raise RuntimeError(f"gmod_special_enum fields must be dict in {fqname}")

        values: dict[str, EnumValue] = {}

        for item in fields_node.items:
            if not isinstance(item, PyIRDictItem):
                continue

            if not isinstance(item.key, PyIRConstant):
                continue

            if not isinstance(item.value, PyIRTuple):
                continue

            if len(item.value.elements) != 2:
                continue

            name = item.key.value
            if not isinstance(name, str):
                continue

            kind_node = item.value.elements[0]
            val_node = item.value.elements[1]

            if not isinstance(kind_node, PyIRConstant):
                continue

            kind = kind_node.value
            if kind not in ("int", "str", "bool", "global"):
                raise RuntimeError(
                    f"Invalid enum value kind {kind!r} in {fqname}.{name}"
                )

            if not isinstance(val_node, PyIRConstant):
                continue

            values[name] = EnumValue(
                name=name,
                kind=kind,  # type: ignore[arg-type]
                value=val_node.value,  # pyright: ignore[reportArgumentType]
            )

        return EnumInfo(
            class_symbol_id=sid,
            fqname=fqname,
            values=values,
            origin="gmod_special_enum",
        )

    @staticmethod
    def _collect_python_enum(cls_node: PyIRClassDef, sid: int, fqname: str) -> EnumInfo:
        values: dict[str, EnumValue] = {}

        for item in cls_node.body:
            if not isinstance(item, PyIRAssign):
                continue

            if len(item.targets) != 1:
                continue

            tgt = item.targets[0]
            if not isinstance(tgt, PyIRVarUse):
                continue

            if not isinstance(item.value, PyIRConstant):
                continue

            v = item.value.value
            if isinstance(v, bool):
                kind = "bool"

            elif isinstance(v, int):
                kind = "int"

            elif isinstance(v, str):
                kind = "str"

            else:
                continue

            values[tgt.name] = EnumValue(
                name=tgt.name,
                kind=kind,  # type: ignore[arg-type]
                value=v,
            )

        return EnumInfo(
            class_symbol_id=sid,
            fqname=fqname,
            values=values,
            origin="python_enum",
        )

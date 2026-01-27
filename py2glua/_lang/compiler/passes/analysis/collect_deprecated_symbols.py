from __future__ import annotations

from ....py.ir_builder import PyIRFile
from ....py.ir_dataclass import (
    PyIRClassDef,
    PyIRConstant,
    PyIRDecorator,
    PyIRFunctionDef,
    PyIRNode,
)
from .ctx import AnalysisContext, AnalysisPass, DeprecatedSymbolInfo


class CollectDeprecatedSymbolsPass(AnalysisPass):
    @classmethod
    def run(cls, ir: PyIRFile, ctx: AnalysisContext) -> None:
        if ir.path is None:
            return

        cls._walk_block(ir.body, ir=ir, ctx=ctx, owner_class=None)

    @classmethod
    def _walk_block(
        cls,
        body: list[PyIRNode],
        *,
        ir: PyIRFile,
        ctx: AnalysisContext,
        owner_class: PyIRClassDef | None,
    ) -> None:
        for node in body:
            if isinstance(node, PyIRClassDef):
                cls._handle_class(node, ir=ir, ctx=ctx)
                cls._walk_block(node.body, ir=ir, ctx=ctx, owner_class=node)
                continue

            if isinstance(node, PyIRFunctionDef):
                cls._handle_function(node, ir=ir, ctx=ctx, owner_class=owner_class)
                cls._walk_block(node.body, ir=ir, ctx=ctx, owner_class=owner_class)
                continue

    @classmethod
    def _handle_class(
        cls, node: PyIRClassDef, *, ir: PyIRFile, ctx: AnalysisContext
    ) -> None:
        dep = cls._find_deprecated_decorator(node.decorators)
        if dep is None:
            return

        class_sid = cls._resolve_module_class_sid(node.name, ir=ir, ctx=ctx)
        if class_sid is None:
            return

        ctx.deprecated_symbols[class_sid] = DeprecatedSymbolInfo(
            symbol_id=class_sid,
            fqname=cls._fqname(class_sid, ctx),
            message=cls._extract_message(dep),
            file=ir.path,
            line=node.line,
            offset=node.offset,
        )

    @classmethod
    def _handle_function(
        cls,
        node: PyIRFunctionDef,
        *,
        ir: PyIRFile,
        ctx: AnalysisContext,
        owner_class: PyIRClassDef | None,
    ) -> None:
        dep = cls._find_deprecated_decorator(node.decorators)
        if dep is None:
            return

        if owner_class is None:
            fn_sid = cls._resolve_module_function_sid(node.name, ir=ir, ctx=ctx)
            if fn_sid is None:
                return

            ctx.deprecated_symbols[fn_sid] = DeprecatedSymbolInfo(
                symbol_id=fn_sid,
                fqname=cls._fqname(fn_sid, ctx),
                message=cls._extract_message(dep),
                file=ir.path,
                line=node.line,
                offset=node.offset,
            )
            return

        class_sid = cls._resolve_module_class_sid(owner_class.name, ir=ir, ctx=ctx)
        if class_sid is None:
            return

        key = (class_sid, node.name)
        ctx.deprecated_members[key] = DeprecatedSymbolInfo(
            symbol_id=class_sid,
            fqname=f"{cls._fqname(class_sid, ctx)}.{node.name}",
            message=cls._extract_message(dep),
            file=ir.path,
            line=node.line,
            offset=node.offset,
        )

    @classmethod
    def _find_deprecated_decorator(
        cls,
        decorators: list[PyIRDecorator],
    ) -> PyIRDecorator | None:
        for d in decorators:
            if cls._is_deprecated_decorator(d):
                return d

        return None

    @staticmethod
    def _is_deprecated_decorator(dec: PyIRDecorator) -> bool:
        name = getattr(dec, "name", None)
        if not isinstance(name, str):
            return False

        return name == "deprecated" or name.endswith(".deprecated")

    @staticmethod
    def _is_deprecated_callee(expr) -> bool:
        if expr is None:
            return False

        if expr.__class__.__name__ == "PyIRVarUse":
            name = getattr(expr, "name", None)
            return name == "deprecated"

        if expr.__class__.__name__ == "PyIRAttribute":
            dotted = CollectDeprecatedSymbolsPass._expr_to_dotted(expr)
            if dotted is None:
                return False

            return dotted == "deprecated" or dotted.endswith(".deprecated")

        return False

    @staticmethod
    def _expr_to_dotted(expr) -> str | None:
        if expr is None:
            return None

        if expr.__class__.__name__ == "PyIRVarUse":
            return getattr(expr, "name", None)

        if expr.__class__.__name__ == "PyIRAttribute":
            parts: list[str] = []
            cur = expr
            while cur is not None and cur.__class__.__name__ == "PyIRAttribute":
                attr = getattr(cur, "attr", None)
                if not isinstance(attr, str) or not attr:
                    return None
                parts.append(attr)
                cur = getattr(cur, "value", None)

            base = CollectDeprecatedSymbolsPass._expr_to_dotted(cur)
            if base is None:
                return None

            parts.reverse()
            return base + "." + ".".join(parts)

        return None

    @staticmethod
    def _extract_message(dec: PyIRDecorator) -> str | None:
        if not getattr(dec, "args_p", None):
            return None

        a0 = dec.args_p[0]
        if isinstance(a0, PyIRConstant) and isinstance(a0.value, str):
            return a0.value

        return None

    @staticmethod
    def _fqname(sid: int, ctx: AnalysisContext) -> str:
        info = ctx.symbols.get(sid)
        return info.fqname if info else f"<sym:{sid}>"

    @staticmethod
    def _resolve_module_class_sid(
        name: str, *, ir: PyIRFile, ctx: AnalysisContext
    ) -> int | None:
        if ir.path is None:
            return None

        for sid, info in ctx.symbols.items():
            if info.file != ir.path:
                continue

            sym = info.symbol
            if sym.scope == "module" and sym.kind == "class" and sym.name == name:
                return sid

        return None

    @staticmethod
    def _resolve_module_function_sid(
        name: str, *, ir: PyIRFile, ctx: AnalysisContext
    ) -> int | None:
        if ir.path is None:
            return None

        for sid, info in ctx.symbols.items():
            if info.file != ir.path:
                continue

            sym = info.symbol
            if sym.scope == "module" and sym.kind == "function" and sym.name == name:
                return sid

        return None

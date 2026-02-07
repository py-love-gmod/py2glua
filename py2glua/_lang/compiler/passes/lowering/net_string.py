from __future__ import annotations

from pathlib import Path
from typing import Final

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRConstant,
    PyIRDecorator,
    PyIRFile,
    PyIRFunctionDef,
    PyIRNode,
)
from ..analysis.symlinks import PyIRSymLink, SymLinkContext


class CollectRegisterArgDeclsPass:
    """
    Lowering-pass (фаза 1):

    Собирает функции, помеченные декоратором
        @CompilerDirective.register_arg(index=?)

    Записывает в ctx:
      - ctx._register_arg_targets: dict[int, int]
        (symbol_id функции -> индекс аргумента)

      - ctx._register_arg_symid_by_decl_cache:
        dict[(Path, line, name) -> symbol_id]

    Требования:
      - index должен быть целочисленным литералом
      - декоратор должен быть ИМЕННО CompilerDirective.register_arg
      - декоратор вырезается из IR
    """

    _TARGETS_FIELD: Final[str] = "_register_arg_targets"
    _DECL_CACHE_FIELD: Final[str] = "_register_arg_symid_by_decl_cache"

    _CORE_MODULE: Final[str] = "py2glua.glua.core.compiler_directive"
    _CD_EXPORT: Final[str] = "CompilerDirective"
    _DECORATOR_ATTR: Final[str] = "register_arg"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectRegisterArgDeclsPass.run ожидает PyIRFile")

        targets = getattr(ctx, cls._TARGETS_FIELD, None)
        if not isinstance(targets, dict):
            targets = {}
            setattr(ctx, cls._TARGETS_FIELD, targets)

        if ir.path is None:
            return ir

        symid_by_decl = cls._get_symid_by_decl(ctx)
        fp = cls._canon_path(ir.path)
        if fp is None:
            return ir

        cd_canon_id = cls._compiler_directive_canon_id(ctx)

        for node in ir.walk():
            if not isinstance(node, PyIRFunctionDef):
                continue

            if node.line is None:
                continue

            fn_symid = symid_by_decl.get((fp, node.line, node.name))
            if fn_symid is None:
                continue

            new_decorators: list[PyIRDecorator] = []
            for d in node.decorators or []:
                matched, index = cls._match_register_arg_decorator(
                    d, cd_canon_id=cd_canon_id
                )

                if not matched:
                    new_decorators.append(d)
                    continue

                if index < 0:
                    CompilerExit.user_error_node(
                        "CompilerDirective.register_arg: индекс должен быть >= 0",
                        path=ir.path,
                        node=d,
                    )

                targets[int(fn_symid)] = int(index)

            node.decorators = new_decorators

        return ir

    @staticmethod
    def _canon_path(p: Path | None) -> Path | None:
        if p is None:
            return None
        try:
            return p.resolve()
        except Exception:
            return p

    @classmethod
    def _get_symid_by_decl(
        cls, ctx: SymLinkContext
    ) -> dict[tuple[Path, int, str], int]:
        cached = getattr(ctx, cls._DECL_CACHE_FIELD, None)
        if isinstance(cached, dict):
            return cached

        out: dict[tuple[Path, int, str], int] = {}
        for sym in ctx.symbols.values():
            if sym.decl_file is None or sym.decl_line is None:
                continue
            p = cls._canon_path(sym.decl_file)
            if p is None:
                continue
            out[(p, sym.decl_line, sym.name)] = int(sym.id.value)

        setattr(ctx, cls._DECL_CACHE_FIELD, out)
        return out

    @classmethod
    def _compiler_directive_canon_id(cls, ctx: SymLinkContext) -> int | None:
        try:
            ctx.ensure_module_index()
            sym = ctx.get_exported_symbol(cls._CORE_MODULE, cls._CD_EXPORT)
            return int(sym.id.value) if sym is not None else None
        
        except Exception:
            return None

    @classmethod
    def _parse_decorator_call(
        cls, d: PyIRDecorator
    ) -> tuple[PyIRNode, list[PyIRNode], dict[str, PyIRNode]]:
        if isinstance(d.exper, PyIRCall):
            c = d.exper
            return c.func, list(c.args_p or []), dict(c.args_kw or {})
        return d.exper, list(d.args_p or []), dict(d.args_kw or {})

    @classmethod
    def _match_register_arg_decorator(
        cls,
        d: PyIRDecorator,
        *,
        cd_canon_id: int | None,
    ) -> tuple[bool, int]:
        """
        Возвращает (совпало, индекс).
        Индекс по умолчанию = 0.
        """
        func, args_p, args_kw = cls._parse_decorator_call(d)

        if not isinstance(func, PyIRAttribute):
            return False, 0

        if func.attr != cls._DECORATOR_ATTR:
            return False, 0

        if not cls._is_compiler_directive_base(func.value, cd_canon_id):
            return False, 0

        index = 0

        if args_kw:
            bad = [k for k in args_kw.keys() if k != "index"]
            if bad:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: допускается только keyword 'index'",
                    path=None,
                    node=d,
                    show_path=False,
                )

            if "index" in args_kw:
                index = cls._require_int_literal(args_kw["index"], d)

        if args_p:
            if len(args_p) > 1:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: слишком много позиционных аргументов",
                    path=None,
                    node=d,
                    show_path=False,
                )

            if "index" in args_kw:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: нельзя смешивать позиционный и keyword аргумент",
                    path=None,
                    node=d,
                    show_path=False,
                )

            index = cls._require_int_literal(args_p[0], d)

        return True, int(index)

    @staticmethod
    def _require_int_literal(expr: PyIRNode, where: PyIRNode) -> int:
        if not isinstance(expr, PyIRConstant) or not isinstance(expr.value, int):
            CompilerExit.user_error_node(
                "CompilerDirective.register_arg: индекс должен быть целочисленным литералом",
                path=None,
                node=where,
                show_path=False,
            )
        return int(expr.value)

    @staticmethod
    def _is_compiler_directive_base(base: PyIRNode, cd_canon_id: int | None) -> bool:
        if isinstance(base, PyIRSymLink):
            if cd_canon_id is not None and int(base.symbol_id) == int(cd_canon_id):
                return True
            return base.name == "CompilerDirective"

        if type(base).__name__ == "PyIRVarUse":
            return getattr(base, "name", None) == "CompilerDirective"

        return False


class CollectRegisterArgNetStringsPass:
    """
    Lowering-pass (фаза 2):

    Ищет вызовы функций, помеченных register_arg,
    и извлекает строковый литерал из аргумента index.

    Если аргумент:
      - не строковый литерал
      - отсутствует
      - передан через keyword

    → ошибка компиляции.

    Записывает в ctx:
      - ctx._net_string_literals: set[str]
    """

    _TARGETS_FIELD: Final[str] = "_register_arg_targets"
    _NET_SET_FIELD: Final[str] = "_net_string_literals"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectRegisterArgNetStringsPass.run ожидает PyIRFile")

        targets = getattr(ctx, cls._TARGETS_FIELD, None)
        if not isinstance(targets, dict) or not targets:
            return ir

        net_set = getattr(ctx, cls._NET_SET_FIELD, None)
        if not isinstance(net_set, set):
            net_set = set()
            setattr(ctx, cls._NET_SET_FIELD, net_set)

        for node in ir.walk():
            if not isinstance(node, PyIRCall):
                continue

            fn = node.func
            if not isinstance(fn, PyIRSymLink):
                continue

            index = targets.get(int(fn.symbol_id))
            if index is None:
                continue

            if node.args_kw:
                CompilerExit.user_error_node(
                    "register_arg: keyword-аргументы не поддерживаются",
                    path=ir.path,
                    node=node,
                )

            if index >= len(node.args_p):
                CompilerExit.user_error_node(
                    f"register_arg: отсутствует аргумент с индексом {index}",
                    path=ir.path,
                    node=node,
                )

            arg = node.args_p[int(index)]
            if not isinstance(arg, PyIRConstant) or not isinstance(arg.value, str):
                CompilerExit.user_error_node(
                    "register_arg: аргумент должен быть строковым литералом",
                    path=ir.path,
                    node=node,
                )

            net_set.add(arg.value)

        return ir

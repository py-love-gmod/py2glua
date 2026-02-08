from __future__ import annotations

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
from ..common import (
    canon_path,
    collect_core_compiler_directive_local_symbol_ids,
    collect_decl_symid_cache,
    decorator_call_parts,
    is_expr_on_symbol_ids,
)


class CollectRegisterArgDeclsPass:
    """
    Lowering-pass (С„Р°Р·Р° 1).

    РЎРѕР±РёСЂР°РµС‚ С„СѓРЅРєС†РёРё, РїРѕРјРµС‡РµРЅРЅС‹Рµ РґРµРєРѕСЂР°С‚РѕСЂРѕРј:
        @CompilerDirective.register_arg(index=?).

    Р—Р°РїРёСЃС‹РІР°РµС‚ РІ `ctx._register_arg_targets`:
      - `symbol_id` С„СѓРЅРєС†РёРё -> РёРЅРґРµРєСЃ Р°СЂРіСѓРјРµРЅС‚Р°.

    РўСЂРµР±РѕРІР°РЅРёСЏ:
      - `index` РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ С†РµР»РѕС‡РёСЃР»РµРЅРЅС‹Рј Р»РёС‚РµСЂР°Р»РѕРј.
      - РґРµРєРѕСЂР°С‚РѕСЂ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РёРјРµРЅРЅРѕ РёР· core `CompilerDirective.register_arg`.
      - РѕР±СЂР°Р±РѕС‚Р°РЅРЅС‹Р№ РґРµРєРѕСЂР°С‚РѕСЂ РІС‹СЂРµР·Р°РµС‚СЃСЏ РёР· IR.
    """

    _DECL_CACHE_FIELD: Final[str] = "_register_arg_symid_by_decl_cache"
    _DECORATOR_ATTR: Final[str] = "register_arg"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectRegisterArgDeclsPass.run РѕР¶РёРґР°РµС‚ PyIRFile")

        targets = ctx._register_arg_targets

        if ir.path is None:
            return ir

        ctx.ensure_module_index()
        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return ir

        symid_by_decl = collect_decl_symid_cache(ctx, cache_field=cls._DECL_CACHE_FIELD)
        fp = canon_path(ir.path)
        if fp is None:
            return ir

        core_cd_symids = collect_core_compiler_directive_local_symbol_ids(
            ir,
            ctx=ctx,
            current_module=current_module,
        )

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
                    d,
                    core_cd_symids=core_cd_symids,
                    ctx=ctx,
                )

                if not matched:
                    new_decorators.append(d)
                    continue

                if index < 0:
                    CompilerExit.user_error_node(
                        "CompilerDirective.register_arg: РёРЅРґРµРєСЃ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ >= 0",
                        path=ir.path,
                        node=d,
                    )

                targets[int(fn_symid)] = int(index)

            node.decorators = new_decorators

        return ir

    @classmethod
    def _match_register_arg_decorator(
        cls,
        d: PyIRDecorator,
        *,
        core_cd_symids: set[int],
        ctx: SymLinkContext,
    ) -> tuple[bool, int]:
        """
        Р’РѕР·РІСЂР°С‰Р°РµС‚ `(СЃРѕРІРїР°Р»Рѕ, РёРЅРґРµРєСЃ)`.
        РРЅРґРµРєСЃ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ: `0`.
        """
        func, args_p, args_kw = decorator_call_parts(d)

        if not isinstance(func, PyIRAttribute):
            return False, 0

        if func.attr != cls._DECORATOR_ATTR:
            return False, 0

        if not is_expr_on_symbol_ids(
            func.value,
            symbol_ids=core_cd_symids,
            ctx=ctx,
        ):
            return False, 0

        index = 0

        if args_kw:
            bad = [k for k in args_kw.keys() if k != "index"]
            if bad:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: РґРѕРїСѓСЃРєР°РµС‚СЃСЏ С‚РѕР»СЊРєРѕ keyword 'index'",
                    path=None,
                    node=d,
                    show_path=False,
                )

            if "index" in args_kw:
                index = cls._require_int_literal(args_kw["index"], d)

        if args_p:
            if len(args_p) > 1:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: СЃР»РёС€РєРѕРј РјРЅРѕРіРѕ РїРѕР·РёС†РёРѕРЅРЅС‹С… Р°СЂРіСѓРјРµРЅС‚РѕРІ",
                    path=None,
                    node=d,
                    show_path=False,
                )

            if "index" in args_kw:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: РЅРµР»СЊР·СЏ СЃРјРµС€РёРІР°С‚СЊ РїРѕР·РёС†РёРѕРЅРЅС‹Р№ Рё keyword Р°СЂРіСѓРјРµРЅС‚",
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
                "CompilerDirective.register_arg: РёРЅРґРµРєСЃ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ С†РµР»РѕС‡РёСЃР»РµРЅРЅС‹Рј Р»РёС‚РµСЂР°Р»РѕРј",
                path=None,
                node=where,
                show_path=False,
            )
        return int(expr.value)


class CollectRegisterArgNetStringsPass:
    """
    Lowering-pass (С„Р°Р·Р° 2).

    РС‰РµС‚ РІС‹Р·РѕРІС‹ С„СѓРЅРєС†РёР№, РїРѕРјРµС‡РµРЅРЅС‹С… `register_arg`,
    Рё РёР·РІР»РµРєР°РµС‚ СЃС‚СЂРѕРєРѕРІС‹Р№ Р»РёС‚РµСЂР°Р» РёР· Р°СЂРіСѓРјРµРЅС‚Р° СЃ Р·Р°РґР°РЅРЅС‹Рј РёРЅРґРµРєСЃРѕРј.

    Р•СЃР»Рё Р°СЂРіСѓРјРµРЅС‚:
      - РЅРµ СЃС‚СЂРѕРєРѕРІС‹Р№ Р»РёС‚РµСЂР°Р»,
      - РѕС‚СЃСѓС‚СЃС‚РІСѓРµС‚,
      - РїРµСЂРµРґР°РЅ С‡РµСЂРµР· keyword,
    С‚Рѕ СЌС‚Рѕ РѕС€РёР±РєР° РєРѕРјРїРёР»СЏС†РёРё.

    Р—Р°РїРёСЃС‹РІР°РµС‚ СЃС‚СЂРѕРєРё РІ `ctx._net_string_literals`.
    """

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError(
                "CollectRegisterArgNetStringsPass.run РѕР¶РёРґР°РµС‚ PyIRFile"
            )

        targets = ctx._register_arg_targets
        if not targets:
            return ir

        net_set = ctx._net_string_literals

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
                    "register_arg: keyword-Р°СЂРіСѓРјРµРЅС‚С‹ РЅРµ РїРѕРґРґРµСЂР¶РёРІР°СЋС‚СЃСЏ",
                    path=ir.path,
                    node=node,
                )

            if index >= len(node.args_p):
                CompilerExit.user_error_node(
                    f"register_arg: РѕС‚СЃСѓС‚СЃС‚РІСѓРµС‚ Р°СЂРіСѓРјРµРЅС‚ СЃ РёРЅРґРµРєСЃРѕРј {index}",
                    path=ir.path,
                    node=node,
                )

            arg = node.args_p[int(index)]
            if not isinstance(arg, PyIRConstant) or not isinstance(arg.value, str):
                CompilerExit.user_error_node(
                    "register_arg: Р°СЂРіСѓРјРµРЅС‚ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ СЃС‚СЂРѕРєРѕРІС‹Рј Р»РёС‚РµСЂР°Р»РѕРј",
                    path=ir.path,
                    node=node,
                )

            net_set.add(arg.value)

        return ir

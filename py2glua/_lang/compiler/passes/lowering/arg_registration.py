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
    Lowering-pass (фаза 1).

    Сканирует функции, помеченные `@CompilerDirective.register_arg(...)`,
    и записывает в `ctx._register_arg_targets`:
      - `symbol_id` функции -> `(index, reg_type)`.

    Важно:
      - `index` должен быть целочисленным литералом >= 0;
      - `reg_type` обязан быть указан явно;
      - поддерживаются только `net`, `material`, `model`;
      - обработанный декоратор удаляется из IR.
    """

    _DECL_CACHE_FIELD: Final[str] = "_register_arg_symid_by_decl_cache"
    _DECORATOR_ATTR: Final[str] = "register_arg"
    _ALLOWED_REG_TYPES: Final[frozenset[str]] = frozenset(
        {
            "net",
            "material",
            "model",
        }
    )

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectRegisterArgDeclsPass.run ожидает PyIRFile")

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
                matched, index, reg_type = cls._match_register_arg_decorator(
                    d,
                    core_cd_symids=core_cd_symids,
                    ctx=ctx,
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

                targets[int(fn_symid)] = (int(index), reg_type)

            node.decorators = new_decorators

        return ir

    @classmethod
    def _match_register_arg_decorator(
        cls,
        d: PyIRDecorator,
        *,
        core_cd_symids: set[int],
        ctx: SymLinkContext,
    ) -> tuple[bool, int, str]:
        """Возвращает `(matched, index, reg_type)` для register_arg-декоратора."""
        func, args_p, args_kw = decorator_call_parts(d)

        if not isinstance(func, PyIRAttribute):
            return False, 0, ""

        if func.attr != cls._DECORATOR_ATTR:
            return False, 0, ""

        if not is_expr_on_symbol_ids(
            func.value,
            symbol_ids=core_cd_symids,
            ctx=ctx,
        ):
            return False, 0, ""

        index = 0
        reg_type: str | None = None

        if args_kw:
            bad = [k for k in args_kw.keys() if k not in {"index", "reg_type"}]
            if bad:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: допускаются только keyword 'index' и 'reg_type'",
                    path=None,
                    node=d,
                    show_path=False,
                )

            if "index" in args_kw:
                index = cls._require_int_literal(args_kw["index"], d)
                
            if "reg_type" in args_kw:
                reg_type = cls._require_reg_type_literal(args_kw["reg_type"], d)

        if args_p:
            if len(args_p) > 2:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: слишком много позиционных аргументов",
                    path=None,
                    node=d,
                    show_path=False,
                )

            if len(args_p) >= 1 and "index" in args_kw:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: нельзя смешивать позиционный и keyword аргумент 'index'",
                    path=None,
                    node=d,
                    show_path=False,
                )

            if len(args_p) >= 2 and "reg_type" in args_kw:
                CompilerExit.user_error_node(
                    "CompilerDirective.register_arg: нельзя смешивать позиционный и keyword аргумент 'reg_type'",
                    path=None,
                    node=d,
                    show_path=False,
                )

            if len(args_p) >= 1:
                index = cls._require_int_literal(args_p[0], d)
            if len(args_p) >= 2:
                reg_type = cls._require_reg_type_literal(args_p[1], d)

        if reg_type is None:
            CompilerExit.user_error_node(
                "CompilerDirective.register_arg: reg_type должен быть указан явно",
                path=None,
                node=d,
                show_path=False,
            )

        return True, int(index), reg_type

    @staticmethod
    def _require_int_literal(expr: PyIRNode, where: PyIRNode) -> int:
        if not isinstance(expr, PyIRConstant) or not isinstance(expr.value, int):
            CompilerExit.user_error_node(
                "CompilerDirective.register_arg: index должен быть целочисленным литералом",
                path=None,
                node=where,
                show_path=False,
            )
        return int(expr.value)

    @classmethod
    def _require_reg_type_literal(cls, expr: PyIRNode, where: PyIRNode) -> str:
        if not isinstance(expr, PyIRConstant) or not isinstance(expr.value, str):
            CompilerExit.user_error_node(
                "CompilerDirective.register_arg: reg_type должен быть строковым литералом",
                path=None,
                node=where,
                show_path=False,
            )

        value = expr.value.strip().lower()
        if value not in cls._ALLOWED_REG_TYPES:
            CompilerExit.user_error_node(
                "CompilerDirective.register_arg: reg_type должен быть одним из: 'net', 'material', 'model'",
                path=None,
                node=where,
                show_path=False,
            )

        return value


class CollectRegisterArgNetStringsPass:
    """
    Lowering-pass (фаза 2).

    Ищет вызовы функций, помеченных `register_arg`, извлекает строковый
    аргумент по индексу и складывает значение в
    `ctx._register_arg_values_by_type[reg_type]`.

    Формат хранилища:
      - ключ: `reg_type`
      - значение: `list[str]` (без дублей, в порядке появления)
    """

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectRegisterArgNetStringsPass.run ожидает PyIRFile")

        targets = ctx._register_arg_targets
        if not targets:
            return ir

        values_by_type = ctx._register_arg_values_by_type

        for node in ir.walk():
            if not isinstance(node, PyIRCall):
                continue

            fn = node.func
            if not isinstance(fn, PyIRSymLink):
                continue

            target = targets.get(int(fn.symbol_id))
            if target is None:
                continue

            index, reg_type = target

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
                    "register_arg: выбранный аргумент должен быть строковым литералом",
                    path=ir.path,
                    node=node,
                )

            bucket = values_by_type.setdefault(reg_type, [])
            if arg.value not in bucket:
                bucket.append(arg.value)

        return ir

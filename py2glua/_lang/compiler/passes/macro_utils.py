from typing import TYPE_CHECKING, Callable, Iterable, Type

from ...py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRNode,
    PyIRWhile,
    PyIRWith,
)

if TYPE_CHECKING:
    from .analysis.symlinks import PyIRSymLink, SymLinkContext


class MacroUtils:
    """
    Утилиты для macro-pass'ов CompilerDirective.

    Содержит:
      - резолв класса CompilerDirective
      - строгий детектор @CompilerDirective.<kind>[()]
      - общий lifecycle-хелпер для pass-контекста
      - шаблон сбора помеченных функций/классов
    """

    MODULE = "py2glua.glua.core.compiler_directive"
    CLASS = "CompilerDirective"
    CTX_KEY = "_compiler_directive_class_sym_id"

    @classmethod
    def class_sym_id(cls, ctx: SymLinkContext) -> int | None:
        """
        Получить symbol_id класса CompilerDirective.

        Делает:
          - резолв через ctx.get_exported_symbol(...)
          - кэширует результат в ctx

        :param ctx: SymLinkContext текущей компиляции
        :return: symbol_id класса CompilerDirective или None,
                 если класс недоступен
        """
        cached = getattr(ctx, cls.CTX_KEY, None)
        if isinstance(cached, int) or (cached is None and hasattr(ctx, cls.CTX_KEY)):
            return cached

        sym = ctx.get_exported_symbol(cls.MODULE, cls.CLASS)
        sym_id = sym.id.value if sym else None
        setattr(ctx, cls.CTX_KEY, sym_id)
        return sym_id

    @classmethod
    def decorator_kind(
        cls,
        expr: PyIRNode,
        *,
        ctx: SymLinkContext,
        kind: str,
    ) -> bool:
        """
        Проверяет, является ли выражение декоратором вида:

            @CompilerDirective.<kind>
            @CompilerDirective.<kind>()

        где CompilerDirective - экспортированный класс из MODULE.

        :param expr: IR-нода декоратора
        :param ctx: SymLinkContext
        :param kind: имя директивы (например: "anonymous", "contextmanager")
        :return: True, если декоратор строго соответствует указанной директиве
        """
        cls_id = cls.class_sym_id(ctx)
        if cls_id is None:
            return False

        # @CompilerDirective.<kind>
        if isinstance(expr, PyIRAttribute):
            return (
                expr.attr == kind
                and isinstance(expr.value, PyIRSymLink)
                and expr.value.symbol_id == cls_id
            )

        # @CompilerDirective.<kind>()
        if isinstance(expr, PyIRCall):
            f = expr.func
            return (
                isinstance(f, PyIRAttribute)
                and f.attr == kind
                and isinstance(f.value, PyIRSymLink)
                and f.value.symbol_id == cls_id
            )

        return False

    @classmethod
    def ensure_pass_ctx(cls, ctx: SymLinkContext, **defaults) -> None:
        """
        Гарантирует наличие служебных атрибутов у ctx.

        Используется macro-pass'ами для хранения:
          - seen paths
          - done flags
          - collected symbols
          - вспомогательных структур

        :param ctx: SymLinkContext
        :param defaults: пары имя_атрибута -> значение_по_умолчанию
        """
        for key, value in defaults.items():
            if not hasattr(ctx, key):
                setattr(ctx, key, value)

    @classmethod
    def ready_to_finalize(
        cls,
        ctx: SymLinkContext,
        *,
        seen_key: str,
        done_key: str,
    ) -> bool:
        """
        Проверяет, что все файлы проекта уже обработаны,
        и pass можно финализировать.

        Логика:
          - если done_key уже True → False
          - если seen != all_paths → False
          - иначе помечает done_key=True и возвращает True

        :param ctx: SymLinkContext
        :param seen_key: имя ctx-атрибута с set[Path]
        :param done_key: имя ctx-атрибута-флага завершения
        :return: True, если пора вызывать финализацию pass'а
        """
        if getattr(ctx, done_key):
            return False

        all_paths = {p.resolve() for p in getattr(ctx, "_all_paths", set()) if p}
        seen = getattr(ctx, seen_key)

        if all_paths and seen != all_paths:
            return False

        setattr(ctx, done_key, True)
        return True

    @classmethod
    def collect_marked_defs(
        cls,
        ir: PyIRFile,
        ctx: SymLinkContext,
        *,
        node_type: Type,
        kind: str,
        out: dict,
        make_info: Callable[[object, int], object],
        get_symbol_id: Callable[[object, SymLinkContext], int | None],
    ) -> None:
        """
        Универсальный сбор функций или классов,
        помеченных @CompilerDirective.<kind>.

        :param ir: IR-файл
        :param ctx: SymLinkContext
        :param node_type: PyIRFunctionDef или PyIRClassDef
        :param kind: имя директивы
        :param out: словарь sym_id -> info
        :param make_info: фабрика info-объекта
        :param get_symbol_id: функция получения symbol_id объявления
        """
        for node in ir.walk():
            if not isinstance(node, node_type):
                continue

            for dec in getattr(node, "decorators", []):
                expr = getattr(dec, "exper", None)
                if not expr:
                    continue

                if cls.decorator_kind(expr, ctx=ctx, kind=kind):
                    sym_id = get_symbol_id(node, ctx)
                    if sym_id is not None and sym_id not in out:
                        out[sym_id] = make_info(node, sym_id)

                    break

    @classmethod
    def collect_marked_defs_multi(
        cls,
        ir: PyIRFile,
        ctx: SymLinkContext,
        *,
        node_type: type,
        kinds: Iterable[str],
        out: dict,
        make_info,
        get_symbol_id,
    ) -> None:
        """
        Универсальный сбор функций или классов,
        помеченных несколькими @CompilerDirective.<kind>.

        В отличие от collect_marked_defs, позволяет:
          - обрабатывать несколько директив одновременно
          - собрать все найденные kind'ы для одного объявления

        :param ir: IR-файл
        :param ctx: SymLinkContext
        :param node_type: PyIRFunctionDef или PyIRClassDef
        :param kinds: iterable имён директив (например {"inline", "anonymous"})
        :param out: словарь sym_id -> info
        :param make_info: фабрика info-объекта (node, kinds, sym_id) -> object
        :param get_symbol_id: функция получения symbol_id объявления
        """
        kinds_set: set[str] = set(kinds)

        for node in ir.walk():
            if not isinstance(node, node_type):
                continue

            found: set[str] = set()

            for dec in getattr(node, "decorators", ()):
                expr = getattr(dec, "exper", None)
                if not expr:
                    continue

                for k in kinds_set:
                    if cls.decorator_kind(expr, ctx=ctx, kind=k):
                        found.add(k)

            if not found:
                continue

            sym_id = get_symbol_id(node, ctx)
            if sym_id is None or sym_id in out:
                continue

            out[sym_id] = make_info(node, found, sym_id)

    @classmethod
    def statement_lists(cls, node: PyIRNode) -> Iterable[list[PyIRNode]]:
        """
        Возвращает все statement-list'ы, содержащиеся в IR-ноде.

        Используется macro-pass'ами для рекурсивного обхода и переписывания
        тела программы без знания конкретного типа управляющей конструкции.

        :param node: IR-нода, потенциально содержащая statement-list'ы
        :return: Итерируемый набор списков PyIRNode, представляющих тела узла
        :rtype: Iterable[list[PyIRNode]]
        """
        if isinstance(
            node,
            (
                PyIRFile,
                PyIRFunctionDef,
                PyIRClassDef,
                PyIRWhile,
                PyIRFor,
                PyIRWith,
            ),
        ):
            yield node.body

        elif isinstance(node, PyIRIf):
            yield node.body
            yield node.orelse

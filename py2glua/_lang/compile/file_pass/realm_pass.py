"""Модуль, отвечающий за волшебную переменную __realm__ в топ-скоупе PyIRFile.

Я не очень хочу поддерживать старую традицую определения скоупа через имя файла

Это может быть как хуже, так и лучше для сообщества. Не знаю

Разрешён ровно один паттерн в модуле:
    __realm__ = Realm.SERVER
    __realm__ = Realm.CLIENT
    __realm__ = Realm.SHARED

Переменная __realm__ может существовать только как это одно присваивание в верхнем уровне файла.

Во всех остальных местах (чтение / запись / внутри функций и т.п.) её использование запрещено и считается ошибкой.

После обработки:
- в file.context.meta["realm"] записывается строка "SERVER"/"CLIENT"/"SHARED";
- само присваивание удаляется из file.body;
- имя '__realm__' удаляется из file.context.scope_name;
- если присваивания нет, realm по умолчанию = "SHARED".
"""

from ...py.py_ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRFile,
    PyIRVarUse,
)


class RealmPass:
    REALM_NAMES = {"SERVER", "CLIENT", "SHARED"}
    MAGIC_NAME = "__realm__"

    @staticmethod
    def run(file: PyIRFile) -> None:
        ctx = file.context

        realm_value: str | None = None
        assign_node: PyIRAssign | None = None
        assign_target: PyIRVarUse | None = None
        assign_index: int | None = None

        for idx, node in enumerate(file.body):
            if isinstance(node, PyIRAssign):
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], PyIRVarUse)
                    and node.targets[0].name == RealmPass.MAGIC_NAME
                ):
                    if assign_node is not None:
                        raise SyntaxError(
                            "__realm__ may be assigned only once at module top-scope"
                        )

                    realm_value = RealmPass._extract_realm(node)
                    assign_node = node
                    assign_target = node.targets[0]
                    assign_index = idx

        magic_usages: list[PyIRVarUse] = []
        for ir_node in file.walk():
            if isinstance(ir_node, PyIRVarUse) and ir_node.name == RealmPass.MAGIC_NAME:
                magic_usages.append(ir_node)

        if assign_target is not None:
            extra = [n for n in magic_usages if n is not assign_target]
            if extra:
                raise SyntaxError(
                    "__realm__ may only be used in a single top-level assignment; "
                    "its use in any other place is forbidden"
                )

        else:
            if magic_usages:
                raise SyntaxError(
                    "__realm__ is reserved for top-level realm definition and "
                    "cannot be used anywhere else"
                )

        if realm_value is not None:
            ctx.meta["realm"] = realm_value

        else:
            ctx.meta.setdefault("realm", "SHARED")

        if assign_index is not None:
            del file.body[assign_index]

            if hasattr(ctx, "scope_name"):
                try:
                    ctx.scope_name.discard(RealmPass.MAGIC_NAME)

                except Exception:
                    pass

    @staticmethod
    def _extract_realm(assign_node: PyIRAssign) -> str:
        val = assign_node.value

        if (
            isinstance(val, PyIRAttribute)
            and isinstance(val.value, PyIRVarUse)
            and val.value.name == "Realm"
            and val.attr in RealmPass.REALM_NAMES
        ):
            return val.attr

        raise SyntaxError(
            "Invalid __realm__ assignment\n"
            "expected:\n"
            "__realm__ = Realm.SERVER|Realm.CLIENT|Realm.SHARED"
        )

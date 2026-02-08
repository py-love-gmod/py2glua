from __future__ import annotations

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    FileRealm,
    PyIRAnnotatedAssign,
    PyIRAssign,
    PyIRAttribute,
    PyIRConstant,
    PyIRFile,
    PyIRNode,
    PyIRVarUse,
)

_ALLOWED_REALMS = {
    "server",
    "client",
    "menu",
    "shared",
}

_REALM_BY_NAME: dict[str, FileRealm] = {
    "server": FileRealm.SERVER,
    "client": FileRealm.CLIENT,
    "menu": FileRealm.MENU,
    "shared": FileRealm.SHARED,
}


class RealmDirectiveSanityCheckPass:
    """
    Проверяет корректность директивы __realm__.

    Правила:
      - __realm__ можно задавать только на уровне файла
      - __realm__ можно задать только один раз
      - __realm__ нельзя использовать как обычную переменную
      - значение должно быть строковым литералом или атрибутом (*.NAME)
      - допустимые значения: SERVER, CLIENT, MENU, SHARED (без учёта регистра)
    """

    @staticmethod
    def run(ir: PyIRFile) -> None:
        top_level_nodes = list(ir.body)

        realm_assign_node: PyIRNode | None = None
        realm_name_final: str | None = None
        realm_target_ids = (
            RealmDirectiveSanityCheckPass._collect_realm_assign_target_ids(ir)
        )

        for node in ir.walk():
            if not isinstance(node, PyIRVarUse):
                continue
            if node.name != "__realm__":
                continue
            if id(node) in realm_target_ids:
                continue

            CompilerExit.user_error_node(
                "Переменную __realm__ нельзя использовать в выражениях",
                ir.path,
                node,
            )

        for node in ir.walk():
            if not RealmDirectiveSanityCheckPass._is_realm_assign(node):
                continue

            if node not in top_level_nodes:
                CompilerExit.user_error_node(
                    "Переменная __realm__ должна задаваться только на уровне файла",
                    ir.path,
                    node,
                )

            if realm_assign_node is not None:
                CompilerExit.user_error_node(
                    "Переменная __realm__ может быть задана только один раз",
                    ir.path,
                    node,
                )

            realm_assign_node = node
            value = RealmDirectiveSanityCheckPass._assign_value(node)
            realm_name = RealmDirectiveSanityCheckPass._extract_realm_value(value)
            if realm_name is None:
                CompilerExit.user_error_node(
                    "__realm__ должен быть строковым литералом "
                    "или обращением к атрибуту (например: Realm.SERVER)",
                    ir.path,
                    node,
                )

            if realm_name.lower() not in _ALLOWED_REALMS:
                CompilerExit.user_error_node(
                    "__realm__ может принимать только значения: "
                    "SERVER, CLIENT, MENU, SHARED",
                    ir.path,
                    node,
                )

            realm_name_final = realm_name.lower()

        if realm_name_final is not None:
            ir.realm = _REALM_BY_NAME[realm_name_final]
            ir.body = [
                st
                for st in ir.body
                if not RealmDirectiveSanityCheckPass._is_realm_assign(st)
            ]

    @staticmethod
    def _collect_realm_assign_target_ids(ir: PyIRFile) -> set[int]:
        out: set[int] = set()
        for node in ir.walk():
            if not isinstance(node, PyIRAssign):
                continue
            if len(node.targets) != 1:
                continue
            t = node.targets[0]
            if isinstance(t, PyIRVarUse) and t.name == "__realm__":
                out.add(id(t))
        return out

    @staticmethod
    def _is_realm_assign(node: PyIRNode) -> bool:
        if isinstance(node, PyIRAssign):
            if len(node.targets) != 1:
                return False

            t = node.targets[0]
            return isinstance(t, PyIRVarUse) and t.name == "__realm__"

        if isinstance(node, PyIRAnnotatedAssign):
            return node.name == "__realm__"

        return False

    @staticmethod
    def _assign_value(node: PyIRNode) -> PyIRNode:
        if isinstance(node, PyIRAssign):
            return node.value
        if isinstance(node, PyIRAnnotatedAssign):
            return node.value

        raise TypeError("_assign_value ожидает узел присваивания __realm__")

    @staticmethod
    def _extract_realm_value(node: PyIRNode) -> str | None:
        if isinstance(node, PyIRConstant) and isinstance(node.value, str):
            return node.value

        if isinstance(node, PyIRAttribute):
            return node.attr

        return None

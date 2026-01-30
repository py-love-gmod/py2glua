from __future__ import annotations

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
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
        realm_assign_node: PyIRAssign | None = None

        for node in ir.walk():
            if isinstance(node, PyIRVarUse) and node.name == "__realm__":
                CompilerExit.user_error_node(
                    "Переменную __realm__ нельзя использовать в выражениях",
                    ir.path,
                    node,
                )

            if not isinstance(node, PyIRAssign):
                continue

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

            realm_name = RealmDirectiveSanityCheckPass._extract_realm_value(node.value)
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

    @staticmethod
    def _is_realm_assign(node: PyIRAssign) -> bool:
        if len(node.targets) != 1:
            return False

        t = node.targets[0]
        return isinstance(t, PyIRVarUse) and t.name == "__realm__"

    @staticmethod
    def _extract_realm_value(node: PyIRNode) -> str | None:
        if isinstance(node, PyIRConstant) and isinstance(node.value, str):
            return node.value

        if isinstance(node, PyIRAttribute):
            return node.attr

        return None

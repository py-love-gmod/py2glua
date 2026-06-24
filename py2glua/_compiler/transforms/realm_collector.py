from __future__ import annotations

from _utils import Shutdown, logger
from plg_reader import (
    IRAnnotatedAssign,
    IRAssign,
    IRAttribute,
    IRConstant,
    IRName,
    IRNode,
)

from ..dt import CompilationUnit
from ..symbol_table import SymbolRefIR

VALID_REALMS = frozenset({"shared", "client", "server", "menu", "stub"})
_VALID_REALMS_STR = ", ".join(sorted(VALID_REALMS)).upper()


class RealmCollector:
    @staticmethod
    def collect(unit: CompilationUnit) -> CompilationUnit:
        for rel_path, ir_file in unit.irs.items():
            realm = None
            first_realm_pos = None
            new_body = []

            for stmt in ir_file.body:
                if not _is_realm_assignment(stmt):
                    new_body.append(stmt)

                else:
                    if realm is not None:
                        logger.warning(
                            "Дубликат __realm__ в %s:%d:%d (первое объявление на %d:%d) - "
                            "будет использовано первое значение",
                            rel_path,
                            stmt.pos[0],
                            stmt.pos[1],
                            first_realm_pos[0] if first_realm_pos else 0,
                            first_realm_pos[1] if first_realm_pos else 0,
                        )

                    else:
                        realm = _extract_realm(stmt, rel_path)
                        first_realm_pos = stmt.pos

            if realm is None:
                realm = "shared"

            unit.realm_dict[rel_path] = realm
            ir_file.body = new_body

        return unit


def _get_target_name(node: IRNode) -> str | None:
    if isinstance(node, IRName):
        return node.name

    if isinstance(node, SymbolRefIR):
        return node.symbol.name

    return None


def _is_realm_assignment(node: IRNode) -> bool:
    if isinstance(node, IRAssign):
        return (
            len(node.targets) == 1 and _get_target_name(node.targets[0]) == "__realm__"
        )

    if isinstance(node, IRAnnotatedAssign):
        return _get_target_name(node.target) == "__realm__"

    return False


def _extract_realm(node: IRNode, rel_path: str) -> str:
    value_node = node.value if isinstance(node, (IRAssign, IRAnnotatedAssign)) else None
    if value_node is None:
        Shutdown.user_error(
            f"Нет значения для __realm__ в {rel_path}:{node.pos[0]}:{node.pos[1]}"
        )

    if isinstance(value_node, IRConstant) and isinstance(value_node.value, str):
        raw = value_node.value
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
            raw = raw[1:-1]

        raw_lower = raw.lower()
        if raw_lower in VALID_REALMS:
            return raw_lower

        Shutdown.user_error(
            f"Недопустимое значение __realm__: '{raw}'. "
            f"Допустимые: {_VALID_REALMS_STR} в {rel_path}:{value_node.pos[0]}:{value_node.pos[1]}"
        )

    if isinstance(value_node, SymbolRefIR):
        sym = value_node.symbol
        if (
            sym.kind == "variable"
            and isinstance(sym.node, IRConstant)
            and isinstance(sym.node.value, str)
        ):
            raw = sym.node.value
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
                raw = raw[1:-1]

            raw_lower = raw.lower()
            if raw_lower in VALID_REALMS:
                return raw_lower

        if sym.name.lower() in VALID_REALMS:
            return sym.name.lower()

        Shutdown.user_error(
            f"Значение __realm__ должно быть строкой или элементом Realm ({_VALID_REALMS_STR}), "
            f"но получен '{sym.name}' в {rel_path}:{value_node.pos[0]}:{value_node.pos[1]}"
        )

    if isinstance(value_node, IRAttribute):
        chain = []
        cur = value_node
        while isinstance(cur, IRAttribute):
            chain.append(cur.attr)
            cur = cur.value

        if isinstance(cur, SymbolRefIR):
            chain.append(cur.symbol.name)

        elif isinstance(cur, IRName):
            chain.append(cur.name)

        else:
            Shutdown.user_error(
                f"Не удалось распознать значение __realm__ в {rel_path}:{value_node.pos[0]}:{value_node.pos[1]}"
            )

        chain.reverse()
        if len(chain) >= 2 and chain[-2] == "Realm":
            member = chain[-1].lower()
            if member in VALID_REALMS:
                return member

        Shutdown.user_error(
            f"Не удалось распознать значение __realm__ в {rel_path}:{value_node.pos[0]}:{value_node.pos[1]}"
        )

    Shutdown.user_error(
        f"Неподдерживаемый тип значения для __realm__ в {rel_path}:{value_node.pos[0]}:{value_node.pos[1]}"
    )

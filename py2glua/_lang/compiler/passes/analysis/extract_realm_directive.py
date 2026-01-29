from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    FileRealm,
    PyIRAssign,
    PyIRAttribute,
    PyIRConstant,
    PyIRFile,
    PyIRVarUse,
)


class ExtractRealmDirectivePass:
    @staticmethod
    def run(ir: PyIRFile, *args) -> None:
        realm_assign: PyIRAssign | None = None

        for node in ir.body:
            if not isinstance(node, PyIRAssign):
                continue

            if len(node.targets) != 1:
                continue

            target = node.targets[0]
            if not isinstance(target, PyIRVarUse):
                continue

            if target.name == "__realm__":
                realm_assign = node
                break

        if realm_assign is None:
            ir.realm = FileRealm.SHARED
            return

        expr = realm_assign.value

        name: str | None = None

        if isinstance(expr, PyIRAttribute):
            name = expr.attr

        elif isinstance(expr, PyIRConstant) and isinstance(expr.value, str):
            name = expr.value

        else:
            CompilerExit.user_error_node(
                "Некорректное значение __realm__",
                ir.path,
                expr,
            )

        key = name.upper()  # type: ignore

        if key not in FileRealm.__members__:
            CompilerExit.user_error_node(
                f"Недопустимое значение __realm__: {name}",
                ir.path,
                expr,
            )

        ir.realm = FileRealm[key]
        ir.body.remove(realm_assign)

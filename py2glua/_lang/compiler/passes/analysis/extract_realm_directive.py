from ....._cli.logging_setup import exit_with_code
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
            exit_with_code(
                1,
                f"Некорректное значение __realm__\n"
                f"Файл: {ir.path}\n"
                f"LINE|OFFSET: {expr.line}|{expr.offset}",
            )

        key = name.upper()  # type: ignore

        if key not in FileRealm.__members__:
            exit_with_code(
                1,
                f"Недопустимое значение __realm__: {name}\n"
                f"Файл: {ir.path}\n"
                f"LINE|OFFSET: {expr.line}|{expr.offset}",
            )

        ir.realm = FileRealm[key]
        ir.body.remove(realm_assign)

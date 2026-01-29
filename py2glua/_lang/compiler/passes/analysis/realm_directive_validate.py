from ....._cli import CompilerExit
from ....py.ir_dataclass import PyIRAssign, PyIRFile, PyIRVarUse


class RealmDirectiveValidatePass:
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

            if target.name != "__realm__":
                continue

            if realm_assign is not None:
                CompilerExit.user_error_node(
                    "__realm__ может быть объявлен только один раз",
                    ir.path,
                    node,
                )

            realm_assign = node

        for node in ir.walk():
            if not isinstance(node, PyIRVarUse):
                continue

            if node.name != "__realm__":
                continue

            if realm_assign is not None and node is realm_assign.targets[0]:
                continue

            CompilerExit.user_error_node(
                "__realm__ является директивой файла и недоступен в коде",
                ir.path,
                node,
            )

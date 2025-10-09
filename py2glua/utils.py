import ast


def is_global_call(value: ast.AST, aliases: dict[str, str], target_attr: str) -> bool:
    if not isinstance(value, ast.Call):
        return False

    func = value.func

    # Global.<target_attr>(...)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base_path = aliases.get(func.value.id, func.value.id)
        return base_path.endswith("py2glua.glua.Global") and func.attr == target_attr

    # glua.Global.<target_attr>(...)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
    ):
        base_path = aliases.get(func.value.value.id, func.value.value.id)
        return (
            base_path.endswith("py2glua.glua")
            and func.value.attr == "Global"
            and func.attr == target_attr
        )

    return False

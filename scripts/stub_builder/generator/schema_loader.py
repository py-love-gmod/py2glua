from .models import (
    Argument,
    ClassDef,
    EnumDef,
    EnumItem,
    FunctionDef,
    Return,
    SchemaEntry,
    StructDef,
)


def _parse_argument(raw: dict) -> Argument:
    callback = None
    if "callback" in raw:
        cb_args = [_parse_argument(a) for a in raw["callback"].get("arguments", [])]
        cb_rets = [_parse_return(r) for r in raw["callback"].get("returns", [])]
        callback = FunctionDef(
            name="",
            arguments=cb_args,
            returns=cb_rets,
        )

    return Argument(
        name=raw.get("name", ""),
        type=raw.get("type", "Any"),
        description=raw.get("description", ""),
        kind=raw.get("kind", "positional"),
        callback=callback,
    )


def _parse_return(raw: dict) -> Return:
    return Return(
        type=raw.get("type", "Any"),
        description=raw.get("description", ""),
    )


def _parse_function(raw: dict, is_static: bool = False) -> FunctionDef:
    args = []
    for item in raw.get("arguments", []):
        if "args" in item:
            for arg in item["args"]:
                parsed = _parse_argument(arg)
                if parsed is not None:
                    args.append(parsed)

        else:
            parsed = _parse_argument(item)
            if parsed is not None:
                args.append(parsed)

    rets = [_parse_return(r) for r in raw.get("returns", [])]
    return FunctionDef(
        name=raw["name"],
        description=raw.get("description", ""),
        arguments=args,
        returns=rets,
        is_static=is_static,
    )


def _parse_class(raw: dict, is_static: bool) -> ClassDef:
    methods_raw = raw.get("methods", [])
    methods = [_parse_function(m, is_static) for m in methods_raw]
    return ClassDef(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        parent=raw.get("parent"),
        methods=methods,
        is_static=is_static,
    )


def _parse_enum(raw: dict) -> EnumDef:
    items_raw = raw.get("items", [])
    items = [
        EnumItem(
            key=item.get("key", ""),
            value=str(item.get("value", "")),
            description=item.get("description", ""),
        )
        for item in items_raw
    ]

    return EnumDef(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        items=items,
    )


def _parse_struct(raw: dict) -> StructDef:
    fields_raw = raw.get("fields", [])
    fields = [_parse_argument(f) for f in fields_raw]
    return StructDef(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        fields=fields,
    )


def load_entries(schema: dict[str, dict]) -> dict[str, SchemaEntry]:
    entries: dict[str, SchemaEntry] = {}
    for raw_name, raw_data in schema.items():
        name = raw_name.replace(".", "_")
        kind = raw_data.get("kind", "class")
        if kind in ("class", "classfunc"):
            entries[name] = _parse_class(raw_data | {"name": name}, is_static=False)

        elif kind in ("library", "libraryfunc"):
            entries[name] = _parse_class(raw_data | {"name": name}, is_static=True)

        elif kind == "enum":
            entries[name] = _parse_enum(raw_data | {"name": name})

        elif kind == "struct":
            entries[name] = _parse_struct(raw_data | {"name": name})

        else:
            print(f"Warning: unsupported kind '{kind}' for {name}, skipping")

    return entries

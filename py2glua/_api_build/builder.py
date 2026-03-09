from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._cli import CompilerExit
from .models import (
    ALLOWED_BODIES,
    ApiArgSpec,
    ApiClassIfSpec,
    ApiClassSpec,
    ApiDecoratorSpec,
    ApiFieldSpec,
    ApiFunctionSpec,
    ApiTypeVarSpec,
)
from .module_builder import render_api_module
from .function_builder import (
    require_identifier,
    require_non_empty_str,
    require_rel_file_path,
    render_python_literal,
)
from .yaml_lite import parse_yaml


@dataclass(slots=True)
class ApiFileSpec:
    module: Path
    imports: list[str]
    type_vars: list[ApiTypeVarSpec]
    functions: list[ApiFunctionSpec]
    classes: list[ApiClassSpec]


@dataclass(slots=True)
class LocMetaSpec:
    code: str
    name: str
    defaults_to_template: str
    deprecated_template: str


def build_api_bundle(*, data_root: Path, build_root: Path, lang: str) -> Path:
    spec = _load_api_spec(data_root=data_root, lang=lang)
    imports = list(spec.imports)

    if spec.type_vars:
        needs_typevar_import = any(tv.expression.startswith("TypeVar(") for tv in spec.type_vars)
        if needs_typevar_import and not any("TypeVar" in imp for imp in imports):
            imports.insert(0, "from typing import TypeVar")

    if _has_any_deprecated(spec.functions, spec.classes):
        dep_import = "from warnings import deprecated"
        if dep_import not in imports:
            imports.insert(0, dep_import)

    build_root.mkdir(parents=True, exist_ok=True)
    out_api = _resolve_under(build_root, spec.module, field_name="api spec: module")
    out_api.parent.mkdir(parents=True, exist_ok=True)
    _ensure_package_inits(build_root, out_api.parent)
    out_api.write_text(
        render_api_module(
            imports=imports,
            type_vars=spec.type_vars,
            functions=spec.functions,
            classes=spec.classes,
        ),
        encoding="utf-8",
    )
    return out_api


def _load_api_spec(*, data_root: Path, lang: str) -> ApiFileSpec:
    api_root = data_root / "api"
    loc_meta = _load_loc_meta(data_root=data_root, lang=lang)
    loc_root = data_root / "loc" / loc_meta.code

    api_files = _collect_yaml_files(
        api_root,
        empty_error=f"Р’ РїР°РїРєРµ API spec РЅРµС‚ .yml/.yaml С„Р°Р№Р»РѕРІ: {api_root}",
        missing_error=f"РќРµ РЅР°Р№РґРµРЅР° РїР°РїРєР° API spec: {api_root}",
    )

    module: Path | None = None
    imports: list[str] = []
    imports_seen: set[str] = set()
    type_vars: list[ApiTypeVarSpec] = []
    type_var_names: set[str] = set()
    functions: list[ApiFunctionSpec] = []
    classes: list[ApiClassSpec] = []

    fn_names: set[str] = set()
    cls_paths: set[str] = set()
    method_paths: set[str] = set()
    field_paths: set[str] = set()

    for path in api_files:
        source_name = path.relative_to(data_root).as_posix()
        raw = _load_yaml_file(path, source_name)
        if not isinstance(raw, dict):
            CompilerExit.user_error(
                f"{source_name}: РєРѕСЂРµРЅСЊ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
                show_path=False,
                show_pos=False,
            )

        _ensure_allowed_api_top_keys(raw, source_name)

        raw_module = raw.get("module")
        if raw_module is None:
            CompilerExit.user_error(
                f"{source_name}: module РѕР±СЏР·Р°С‚РµР»РµРЅ",
                show_path=False,
                show_pos=False,
            )

        cur_module = Path(require_rel_file_path(raw_module, f"{source_name}: module"))
        if module is None:
            module = cur_module
        elif module != cur_module:
            CompilerExit.user_error(
                f"{source_name}: module РєРѕРЅС„Р»РёРєС‚СѓРµС‚ СЃ РґСЂСѓРіРёРј С„Р°Р№Р»РѕРј ({cur_module} != {module})",
                show_path=False,
                show_pos=False,
            )

        raw_imports = raw.get("imports", [])
        if raw_imports is None:
            raw_imports = []
        if not isinstance(raw_imports, list):
            CompilerExit.user_error(
                f"{source_name}: imports РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ СЃРїРёСЃРєРѕРј СЃС‚СЂРѕРє",
                show_path=False,
                show_pos=False,
            )
        for item in raw_imports:
            imp = require_non_empty_str(item, f"{source_name}: imports[]")
            if imp not in imports_seen:
                imports_seen.add(imp)
                imports.append(imp)

        raw_type_vars = raw.get("type_vars", [])
        if raw_type_vars is None:
            raw_type_vars = []
        if not isinstance(raw_type_vars, list):
            CompilerExit.user_error(
                f"{source_name}: type_vars должен быть списком",
                show_path=False,
                show_pos=False,
            )
        for i, item in enumerate(raw_type_vars):
            tv = _parse_type_var(item, idx=i, source_name=source_name)
            if tv.name in type_var_names:
                CompilerExit.user_error(
                    f"{source_name}: дублирующееся имя type var '{tv.name}'",
                    show_path=False,
                    show_pos=False,
                )
            type_var_names.add(tv.name)
            type_vars.append(tv)

        raw_functions = raw.get("functions", [])
        if raw_functions is None:
            raw_functions = []
        if not isinstance(raw_functions, list):
            CompilerExit.user_error(
                f"{source_name}: functions РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ СЃРїРёСЃРєРѕРј",
                show_path=False,
                show_pos=False,
            )
        for i, item in enumerate(raw_functions):
            fn = _parse_function(item, idx=i, source_name=source_name)
            if fn.name in fn_names:
                CompilerExit.user_error(
                    f"{source_name}: РґСѓР±Р»РёСЂСѓСЋС‰Р°СЏСЃСЏ С„СѓРЅРєС†РёСЏ '{fn.name}'",
                    show_path=False,
                    show_pos=False,
                )
            fn_names.add(fn.name)
            functions.append(fn)

        raw_classes = raw.get("classes", [])
        if raw_classes is None:
            raw_classes = []
        if not isinstance(raw_classes, list):
            CompilerExit.user_error(
                f"{source_name}: classes РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ СЃРїРёСЃРєРѕРј",
                show_path=False,
                show_pos=False,
            )
        for i, item in enumerate(raw_classes):
            classes.append(
                _parse_class(
                    item,
                    idx=i,
                    source_name=source_name,
                    parent_path=None,
                    cls_paths=cls_paths,
                    method_paths=method_paths,
                    field_paths=field_paths,
                )
            )

    if module is None:
        CompilerExit.user_error(
            "api spec: РЅРµ РЅР°Р№РґРµРЅ module РЅРё РІ РѕРґРЅРѕРј С„Р°Р№Р»Рµ data/api/**/*.yml",
            show_path=False,
            show_pos=False,
        )
    if not functions and not classes:
        CompilerExit.user_error(
            "api spec: РїСѓСЃС‚Рѕ (РЅРµС‚ functions Рё classes)",
            show_path=False,
            show_pos=False,
        )

    loc_functions, loc_classes, loc_methods, loc_fields = _load_loc_maps(loc_root, data_root)

    _validate_loc_keys(loc_functions, fn_names, "С„СѓРЅРєС†РёРё", loc_meta.code)
    _validate_loc_keys(loc_classes, cls_paths, "РєР»Р°СЃСЃР°", loc_meta.code)
    _validate_loc_keys(loc_methods, method_paths, "РјРµС‚РѕРґР°", loc_meta.code)
    _validate_loc_keys(loc_fields, field_paths, "РїРѕР»СЏ", loc_meta.code)

    for fn in functions:
        loc = loc_functions.get(fn.name)
        payload, source_name = _split_loc(loc, loc_meta.code)
        fn.docstring, fn.deprecated = _build_function_doc_payload(
            entry=payload,
            source_name=source_name,
            fn_name=fn.name,
            args=fn.args,
            returns=fn.returns,
            loc_meta=loc_meta,
        )

    for cls in classes:
        _apply_class_loc(
            cls,
            parent_path=None,
            lang=loc_meta.code,
            loc_meta=loc_meta,
            loc_classes=loc_classes,
            loc_methods=loc_methods,
            loc_fields=loc_fields,
        )

    return ApiFileSpec(
        module=module,
        imports=imports,
        type_vars=type_vars,
        functions=functions,
        classes=classes,
    )


def _ensure_allowed_api_top_keys(raw: dict[str, Any], source_name: str) -> None:
    allowed = {"module", "imports", "type_vars", "functions", "classes"}
    for key in raw:
        if key not in allowed:
            CompilerExit.user_error(
                f"{source_name}: РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РєР»СЋС‡ '{key}'",
                show_path=False,
                show_pos=False,
            )


def _load_loc_meta(*, data_root: Path, lang: str) -> LocMetaSpec:
    source_name = "data/loc_meta.yml"
    path = data_root / "loc_meta.yml"
    if not path.exists() or not path.is_file():
        CompilerExit.user_error(
            f"РќРµ РЅР°Р№РґРµРЅ С„Р°Р№Р» РјРµС‚Р°РґР°РЅРЅС‹С… Р»РѕРєР°Р»РёР·Р°С†РёРё: {path}",
            show_path=False,
            show_pos=False,
        )

    raw = _load_yaml_file(path, source_name)
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: РєРѕСЂРµРЅСЊ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
            show_path=False,
            show_pos=False,
        )

    allowed_top = {"languages"}
    for key in raw:
        if key not in allowed_top:
            CompilerExit.user_error(
                f"{source_name}: РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РєР»СЋС‡ '{key}'",
                show_path=False,
                show_pos=False,
            )

    raw_languages = raw.get("languages")
    if not isinstance(raw_languages, dict) or not raw_languages:
        CompilerExit.user_error(
            f"{source_name}: languages РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РЅРµРїСѓСЃС‚С‹Рј mapping",
            show_path=False,
            show_pos=False,
        )

    requested = require_non_empty_str(lang, "lang").strip().lower()
    normalized: dict[str, LocMetaSpec] = {}

    for code_raw, payload in raw_languages.items():
        code = require_non_empty_str(code_raw, f"{source_name}: languages.<code>").strip().lower()
        if code in normalized:
            CompilerExit.user_error(
                f"{source_name}: РґСѓР±Р»РёСЂСѓСЋС‰РёР№СЃСЏ СЏР·С‹Рє '{code}'",
                show_path=False,
                show_pos=False,
            )
        normalized[code] = _parse_loc_meta_language(
            payload,
            source_name=source_name,
            lang_code=code,
        )

    if requested not in normalized:
        available = ", ".join(sorted(normalized))
        CompilerExit.user_error(
            f"РЇР·С‹Рє '{lang}' РЅРµ РїРѕРґРґРµСЂР¶РёРІР°РµС‚СЃСЏ. Р”РѕСЃС‚СѓРїРЅС‹Рµ СЏР·С‹РєРё: {available}",
            show_path=False,
            show_pos=False,
        )

    return normalized[requested]


def _parse_loc_meta_language(raw: Any, *, source_name: str, lang_code: str) -> LocMetaSpec:
    field = f"{source_name}: languages.{lang_code}"
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{field} РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
            show_path=False,
            show_pos=False,
        )

    allowed = {"name", "edge_cases"}
    for key in raw:
        if key not in allowed:
            CompilerExit.user_error(
                f"{field}: РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РєР»СЋС‡ '{key}'",
                show_path=False,
                show_pos=False,
            )

    name = require_non_empty_str(raw.get("name"), f"{field}.name")
    edge_cases = raw.get("edge_cases")
    if not isinstance(edge_cases, dict):
        CompilerExit.user_error(
            f"{field}.edge_cases РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
            show_path=False,
            show_pos=False,
        )

    allowed_edges = {"defaults_to", "deprecated"}
    for key in edge_cases:
        if key not in allowed_edges:
            CompilerExit.user_error(
                f"{field}.edge_cases: РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РєР»СЋС‡ '{key}'",
                show_path=False,
                show_pos=False,
            )

    defaults_to = require_non_empty_str(edge_cases.get("defaults_to"), f"{field}.edge_cases.defaults_to")
    deprecated = require_non_empty_str(edge_cases.get("deprecated"), f"{field}.edge_cases.deprecated")

    for key, template in {
        "defaults_to": defaults_to,
        "deprecated": deprecated,
    }.items():
        if "{value}" not in template:
            CompilerExit.user_error(
                f"{field}.edge_cases.{key}: С€Р°Р±Р»РѕРЅ РґРѕР»Р¶РµРЅ СЃРѕРґРµСЂР¶Р°С‚СЊ '{{value}}'",
                show_path=False,
                show_pos=False,
            )

    return LocMetaSpec(
        code=lang_code,
        name=name,
        defaults_to_template=defaults_to,
        deprecated_template=deprecated,
    )


def _load_loc_maps(
    loc_root: Path,
    data_root: Path,
) -> tuple[
    dict[str, tuple[Any, str]],
    dict[str, tuple[Any, str]],
    dict[str, tuple[Any, str]],
    dict[str, tuple[Any, str]],
]:
    files = _collect_yaml_files(
        loc_root,
        empty_error=f"Р’ РїР°РїРєРµ Р»РѕРєР°Р»РёР·Р°С†РёРё РЅРµС‚ .yml/.yaml С„Р°Р№Р»РѕРІ: {loc_root}",
        missing_error=f"РќРµ РЅР°Р№РґРµРЅР° РїР°РїРєР° Р»РѕРєР°Р»РёР·Р°С†РёРё: {loc_root}",
    )

    fn_map: dict[str, tuple[Any, str]] = {}
    cls_map: dict[str, tuple[Any, str]] = {}
    method_map: dict[str, tuple[Any, str]] = {}
    field_map: dict[str, tuple[Any, str]] = {}

    for path in files:
        source_name = path.relative_to(data_root).as_posix()
        raw = _load_yaml_file(path, source_name)
        if not isinstance(raw, dict):
            CompilerExit.user_error(
                f"{source_name}: РєРѕСЂРµРЅСЊ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
                show_path=False,
                show_pos=False,
            )

        allowed = {"functions", "classes", "methods", "fields"}
        for key in raw:
            if key not in allowed:
                CompilerExit.user_error(
                    f"{source_name}: РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РєР»СЋС‡ '{key}'",
                    show_path=False,
                    show_pos=False,
                )

        _load_flat_loc_section(raw.get("functions", {}), fn_map, source_name, "functions", dotted=False)
        _load_flat_loc_section(raw.get("classes", {}), cls_map, source_name, "classes", dotted=True)
        _load_flat_loc_section(raw.get("methods", {}), method_map, source_name, "methods", dotted=True)
        _load_flat_loc_section(raw.get("fields", {}), field_map, source_name, "fields", dotted=True)

    return fn_map, cls_map, method_map, field_map


def _load_flat_loc_section(
    raw: Any,
    mapping: dict[str, tuple[Any, str]],
    source_name: str,
    section: str,
    *,
    dotted: bool,
) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: {section} РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
            show_path=False,
            show_pos=False,
        )

    for key_raw, entry in raw.items():
        key = require_non_empty_str(key_raw, f"{source_name}: {section}.<name>")
        if dotted:
            for part in key.split("."):
                require_identifier(part, f"{source_name}: {section}.{key}")
        else:
            require_identifier(key, f"{source_name}: {section}.{key}")

        if not isinstance(entry, (str, dict)):
            CompilerExit.user_error(
                f"{source_name}: {section}.{key} РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ string РёР»Рё mapping",
                show_path=False,
                show_pos=False,
            )
        if key in mapping:
            prev = mapping[key][1]
            CompilerExit.user_error(
                f"{source_name}: РґСѓР±Р»РёСЂСѓСЋС‰Р°СЏСЃСЏ Р»РѕРєР°Р»РёР·Р°С†РёСЏ '{section}.{key}' (СѓР¶Рµ РµСЃС‚СЊ РІ {prev})",
                show_path=False,
                show_pos=False,
            )
        mapping[key] = (entry, source_name)


def _parse_type_var(raw: Any, *, idx: int, source_name: str) -> ApiTypeVarSpec:
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: type_var#{idx} должен быть mapping",
            show_path=False,
            show_pos=False,
        )

    allowed = {
        "name",
        "expr",
        "factory",
        "args",
        "kwargs",
        "bound",
        "constraints",
        "covariant",
        "contravariant",
    }
    for key in raw:
        if key not in allowed:
            CompilerExit.user_error(
                f"{source_name}: type_var#{idx}: неизвестный ключ '{key}'",
                show_path=False,
                show_pos=False,
            )

    name = require_identifier(raw.get("name"), f"{source_name}: type_var#{idx} name")
    expr = raw.get("expr")
    if expr is not None:
        return ApiTypeVarSpec(
            name=name,
            expression=require_non_empty_str(expr, f"{source_name}: type_var {name} expr"),
        )

    factory = require_non_empty_str(raw.get("factory", "TypeVar"), f"{source_name}: type_var {name} factory")

    args_raw = raw.get("args", [])
    if args_raw is None:
        args_raw = []
    if not isinstance(args_raw, list):
        CompilerExit.user_error(
            f"{source_name}: type_var {name}: args должен быть list",
            show_path=False,
            show_pos=False,
        )

    kwargs_raw = raw.get("kwargs", {})
    if kwargs_raw is None:
        kwargs_raw = {}
    if not isinstance(kwargs_raw, dict):
        CompilerExit.user_error(
            f"{source_name}: type_var {name}: kwargs должен быть mapping",
            show_path=False,
            show_pos=False,
        )

    constraints_raw = raw.get("constraints")
    constraints: list[str] = []
    if constraints_raw is not None:
        if not isinstance(constraints_raw, list):
            CompilerExit.user_error(
                f"{source_name}: type_var {name}: constraints должен быть list",
                show_path=False,
                show_pos=False,
            )
        for c_idx, item in enumerate(constraints_raw):
            constraints.append(
                require_non_empty_str(
                    item,
                    f"{source_name}: type_var {name}: constraints[{c_idx}]",
                )
            )

    bound_raw = raw.get("bound")
    bound: str | None = None
    if bound_raw is not None:
        bound = require_non_empty_str(bound_raw, f"{source_name}: type_var {name}: bound")

    if constraints and bound is not None:
        CompilerExit.user_error(
            f"{source_name}: type_var {name}: нельзя задавать одновременно constraints и bound",
            show_path=False,
            show_pos=False,
        )

    call_parts: list[str] = [render_python_literal(name)]
    for arg_idx, item in enumerate(args_raw):
        call_parts.append(
            _render_type_var_value(
                item,
                field_name=f"{source_name}: type_var {name}: args[{arg_idx}]",
            )
        )
    for constraint in constraints:
        call_parts.append(constraint)

    kw_parts: list[str] = []
    kw_seen: set[str] = set()
    for kw_raw, value in kwargs_raw.items():
        kw = require_identifier(kw_raw, f"{source_name}: type_var {name}: kwargs.<key>")
        if kw in kw_seen:
            CompilerExit.user_error(
                f"{source_name}: type_var {name}: дублирующийся kwargs ключ '{kw}'",
                show_path=False,
                show_pos=False,
            )
        kw_seen.add(kw)
        kw_value = _render_type_var_value(
            value,
            field_name=f"{source_name}: type_var {name}: kwargs.{kw}",
        )
        kw_parts.append(f"{kw}={kw_value}")

    if bound is not None:
        if "bound" in kw_seen:
            CompilerExit.user_error(
                f"{source_name}: type_var {name}: bound задан и в kwargs, и отдельным полем",
                show_path=False,
                show_pos=False,
            )
        kw_parts.append(f"bound={bound}")

    for variance_key in ("covariant", "contravariant"):
        if variance_key in raw:
            variance_value = raw.get(variance_key)
            if not isinstance(variance_value, bool):
                CompilerExit.user_error(
                    f"{source_name}: type_var {name}: {variance_key} должен быть bool",
                    show_path=False,
                    show_pos=False,
                )
            if variance_key in kw_seen:
                CompilerExit.user_error(
                    f"{source_name}: type_var {name}: {variance_key} задан и в kwargs, и отдельным полем",
                    show_path=False,
                    show_pos=False,
                )
            kw_parts.append(f"{variance_key}={render_python_literal(variance_value)}")

    expression = f"{factory}({', '.join([*call_parts, *kw_parts])})"
    return ApiTypeVarSpec(name=name, expression=expression)


def _render_type_var_value(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return require_non_empty_str(value, field_name)
    return render_python_literal(value)


def _parse_class(
    raw: Any,
    *,
    idx: int,
    source_name: str,
    parent_path: str | None,
    cls_paths: set[str],
    method_paths: set[str],
    field_paths: set[str],
) -> ApiClassSpec:
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: class#{idx} должен быть mapping",
            show_path=False,
            show_pos=False,
        )

    allowed = {"name", "bases", "decorators", "fields", "methods", "classes", "ifs"}
    for key in raw:
        if key not in allowed:
            CompilerExit.user_error(
                f"{source_name}: class#{idx}: неизвестный ключ '{key}'",
                show_path=False,
                show_pos=False,
            )

    name = require_identifier(raw.get("name"), f"{source_name}: class#{idx} name")
    cls_path = name if parent_path is None else f"{parent_path}.{name}"
    if cls_path in cls_paths:
        CompilerExit.user_error(
            f"{source_name}: дублирующийся класс '{cls_path}'",
            show_path=False,
            show_pos=False,
        )
    cls_paths.add(cls_path)

    raw_bases = raw.get("bases", [])
    if isinstance(raw_bases, str):
        bases = [require_non_empty_str(raw_bases, f"{source_name}: class {cls_path} bases")]
    elif isinstance(raw_bases, list):
        bases = [require_non_empty_str(x, f"{source_name}: class {cls_path} bases[]") for x in raw_bases]
    else:
        CompilerExit.user_error(
            f"{source_name}: class {cls_path} bases должен быть string или list",
            show_path=False,
            show_pos=False,
        )

    decorators = _parse_decorators(raw.get("decorators", []), source_name, cls_path)

    local_field_names: set[str] = set()
    local_method_names: set[str] = set()

    fields = _parse_class_fields(
        raw.get("fields", []),
        source_name=source_name,
        class_path=cls_path,
        field_paths=field_paths,
        local_field_names=local_field_names,
    )
    methods = _parse_class_methods(
        raw.get("methods", []),
        source_name=source_name,
        class_path=cls_path,
        method_paths=method_paths,
        local_method_names=local_method_names,
    )
    nested = _parse_class_nested(
        raw.get("classes", []),
        source_name=source_name,
        class_path=cls_path,
        cls_paths=cls_paths,
        method_paths=method_paths,
        field_paths=field_paths,
    )

    ifs_raw = raw.get("ifs", [])
    if ifs_raw is None:
        ifs_raw = []
    if not isinstance(ifs_raw, list):
        CompilerExit.user_error(
            f"{source_name}: class {cls_path} ifs должен быть list",
            show_path=False,
            show_pos=False,
        )

    ifs: list[ApiClassIfSpec] = []
    for cond_idx, cond_raw in enumerate(ifs_raw):
        ifs.append(
            _parse_class_if(
                cond_raw,
                idx=cond_idx,
                source_name=source_name,
                class_path=cls_path,
                cls_paths=cls_paths,
                method_paths=method_paths,
                field_paths=field_paths,
                local_field_names=local_field_names,
                local_method_names=local_method_names,
            )
        )

    return ApiClassSpec(
        name=name,
        bases=bases,
        decorators=decorators,
        fields=fields,
        methods=methods,
        classes=nested,
        ifs=ifs,
        docstring=None,
        deprecated=None,
    )


def _parse_class_if(
    raw: Any,
    *,
    idx: int,
    source_name: str,
    class_path: str,
    cls_paths: set[str],
    method_paths: set[str],
    field_paths: set[str],
    local_field_names: set[str],
    local_method_names: set[str],
) -> ApiClassIfSpec:
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: class {class_path} if#{idx} должен быть mapping",
            show_path=False,
            show_pos=False,
        )

    allowed = {"condition", "fields", "methods", "classes"}
    for key in raw:
        if key not in allowed:
            CompilerExit.user_error(
                f"{source_name}: class {class_path} if#{idx}: неизвестный ключ '{key}'",
                show_path=False,
                show_pos=False,
            )

    condition = require_non_empty_str(raw.get("condition"), f"{source_name}: class {class_path} if#{idx} condition")

    fields = _parse_class_fields(
        raw.get("fields", []),
        source_name=source_name,
        class_path=class_path,
        field_paths=field_paths,
        local_field_names=local_field_names,
    )
    methods = _parse_class_methods(
        raw.get("methods", []),
        source_name=source_name,
        class_path=class_path,
        method_paths=method_paths,
        local_method_names=local_method_names,
    )
    nested = _parse_class_nested(
        raw.get("classes", []),
        source_name=source_name,
        class_path=class_path,
        cls_paths=cls_paths,
        method_paths=method_paths,
        field_paths=field_paths,
    )

    return ApiClassIfSpec(
        condition=condition,
        fields=fields,
        methods=methods,
        classes=nested,
    )


def _parse_class_fields(
    raw: Any,
    *,
    source_name: str,
    class_path: str,
    field_paths: set[str],
    local_field_names: set[str],
) -> list[ApiFieldSpec]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        CompilerExit.user_error(
            f"{source_name}: class {class_path} fields должен быть list",
            show_path=False,
            show_pos=False,
        )

    out: list[ApiFieldSpec] = []
    for i, item in enumerate(raw):
        field = _parse_field(item, idx=i, source_name=source_name, class_path=class_path)
        if field.name in local_field_names:
            CompilerExit.user_error(
                f"{source_name}: class {class_path}: дублирующееся поле '{field.name}'",
                show_path=False,
                show_pos=False,
            )
        local_field_names.add(field.name)

        field_key = f"{class_path}.{field.name}"
        if field_key in field_paths:
            CompilerExit.user_error(
                f"{source_name}: дублирующееся поле '{field_key}'",
                show_path=False,
                show_pos=False,
            )
        field_paths.add(field_key)
        out.append(field)

    return out


def _parse_class_methods(
    raw: Any,
    *,
    source_name: str,
    class_path: str,
    method_paths: set[str],
    local_method_names: set[str],
) -> list[ApiFunctionSpec]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        CompilerExit.user_error(
            f"{source_name}: class {class_path} methods должен быть list",
            show_path=False,
            show_pos=False,
        )

    out: list[ApiFunctionSpec] = []
    for i, item in enumerate(raw):
        method = _parse_function(item, idx=i, source_name=source_name)
        if method.name in local_method_names:
            CompilerExit.user_error(
                f"{source_name}: class {class_path}: дублирующийся метод '{method.name}'",
                show_path=False,
                show_pos=False,
            )
        local_method_names.add(method.name)

        method_key = f"{class_path}.{method.name}"
        if method_key in method_paths:
            CompilerExit.user_error(
                f"{source_name}: дублирующийся метод '{method_key}'",
                show_path=False,
                show_pos=False,
            )
        method_paths.add(method_key)
        out.append(method)

    return out


def _parse_class_nested(
    raw: Any,
    *,
    source_name: str,
    class_path: str,
    cls_paths: set[str],
    method_paths: set[str],
    field_paths: set[str],
) -> list[ApiClassSpec]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        CompilerExit.user_error(
            f"{source_name}: class {class_path} classes должен быть list",
            show_path=False,
            show_pos=False,
        )

    return [
        _parse_class(
            item,
            idx=i,
            source_name=source_name,
            parent_path=class_path,
            cls_paths=cls_paths,
            method_paths=method_paths,
            field_paths=field_paths,
        )
        for i, item in enumerate(raw)
    ]


def _parse_field(raw: Any, *, idx: int, source_name: str, class_path: str) -> ApiFieldSpec:
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: class {class_path} field#{idx} РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
            show_path=False,
            show_pos=False,
        )

    allowed = {"name", "type", "value"}
    for key in raw:
        if key not in allowed:
            CompilerExit.user_error(
                f"{source_name}: class {class_path} field#{idx}: Р Р…Р ВµР С‘Р В·Р Р†Р ВµРЎРѓРЎвЂљР Р…РЎвЂ№Р в„– Р С”Р В»РЎР‹РЎвЂЎ '{key}'",
                show_path=False,
                show_pos=False,
            )

    name = require_identifier(raw.get("name"), f"{source_name}: class {class_path} field#{idx} name")
    ann = None
    if raw.get("type") is not None:
        ann = require_non_empty_str(raw.get("type"), f"{source_name}: class {class_path} field#{idx} type")
    has_value = "value" in raw
    value = raw.get("value")

    return ApiFieldSpec(
        name=name,
        ann=ann,
        has_value=has_value,
        value=value,
        docstring=None,
    )


def _parse_function(raw: Any, *, idx: int, source_name: str) -> ApiFunctionSpec:
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: функция #{idx} должна быть mapping",
            show_path=False,
            show_pos=False,
        )

    if raw.get("gmod_api") is not None or raw.get("register_arg") is not None:
        CompilerExit.user_error(
            f"{source_name}: функция #{idx}: "
            "используй только decorators[] (kind=gmod_api/register_arg), "
            "отдельные ключи gmod_api/register_arg больше не поддерживаются",
            show_path=False,
            show_pos=False,
        )

    allowed = {"name", "args", "returns", "body", "body_lines", "decorators"}
    for key in raw:
        if key not in allowed:
            CompilerExit.user_error(
                f"{source_name}: функция #{idx}: неизвестный ключ '{key}'",
                show_path=False,
                show_pos=False,
            )

    name = require_identifier(raw.get("name"), f"{source_name}: функция #{idx} name")

    args_raw = raw.get("args", [])
    if not isinstance(args_raw, list):
        CompilerExit.user_error(
            f"{source_name}: функция {name}: args должен быть списком",
            show_path=False,
            show_pos=False,
        )
    args = [_parse_arg(item, fn=name, idx=i, source_name=source_name) for i, item in enumerate(args_raw)]

    returns = None
    if raw.get("returns") is not None:
        returns = require_non_empty_str(raw.get("returns"), f"{source_name}: функция {name} returns")

    body = require_non_empty_str(raw.get("body", "pass"), f"{source_name}: функция {name} body").lower()
    if body == "empty":
        body = "decorator"
    if body not in ALLOWED_BODIES:
        CompilerExit.user_error(
            f"{source_name}: функция {name}: body должен быть одним из {sorted(ALLOWED_BODIES)}",
            show_path=False,
            show_pos=False,
        )

    body_lines: list[str] = []
    body_lines_raw = raw.get("body_lines")
    if body == "raw":
        if body_lines_raw is None:
            CompilerExit.user_error(
                f"{source_name}: функция {name}: для body=raw обязателен body_lines",
                show_path=False,
                show_pos=False,
            )
        if not isinstance(body_lines_raw, list):
            CompilerExit.user_error(
                f"{source_name}: функция {name}: body_lines должен быть списком строк",
                show_path=False,
                show_pos=False,
            )
        for line_idx, line in enumerate(body_lines_raw):
            if not isinstance(line, str):
                CompilerExit.user_error(
                    f"{source_name}: функция {name}: body_lines[{line_idx}] должен быть строкой",
                    show_path=False,
                    show_pos=False,
                )
            body_lines.append(line)
    elif body_lines_raw is not None:
        CompilerExit.user_error(
            f"{source_name}: функция {name}: body_lines можно задавать только для body=raw",
            show_path=False,
            show_pos=False,
        )

    decorators = _parse_decorators(raw.get("decorators", []), source_name, name)

    return ApiFunctionSpec(
        name=name,
        args=args,
        returns=returns,
        body=body,
        decorators=decorators,
        body_lines=body_lines,
        docstring=None,
        deprecated=None,
    )

def _parse_decorators(raw: Any, source_name: str, fn_name: str) -> list[ApiDecoratorSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        CompilerExit.user_error(
            f"{source_name}: {fn_name} decorators РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ СЃРїРёСЃРєРѕРј",
            show_path=False,
            show_pos=False,
        )
    return [_parse_decorator(item, fn=fn_name, idx=i, source_name=source_name) for i, item in enumerate(raw)]


def _parse_arg(raw: Any, *, fn: str, idx: int, source_name: str) -> ApiArgSpec:
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: С„СѓРЅРєС†РёСЏ {fn} arg#{idx}: РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
            show_path=False,
            show_pos=False,
        )

    name = require_identifier(raw.get("name"), f"{source_name}: С„СѓРЅРєС†РёСЏ {fn} arg#{idx} name")
    ann = None
    if raw.get("type") is not None:
        ann = require_non_empty_str(raw.get("type"), f"{source_name}: С„СѓРЅРєС†РёСЏ {fn} arg#{idx} type")

    kind = "normal"
    if raw.get("kind") is not None:
        kind = require_non_empty_str(raw.get("kind"), f"{source_name}: С„СѓРЅРєС†РёСЏ {fn} arg#{idx} kind").lower()
        if kind not in {"posonly", "normal", "kwonly", "vararg", "kwarg"}:
            CompilerExit.user_error(
                f"{source_name}: С„СѓРЅРєС†РёСЏ {fn} arg#{idx}: РЅРµРґРѕРїСѓСЃС‚РёРјС‹Р№ kind '{kind}'",
                show_path=False,
                show_pos=False,
            )

    has_default = "default" in raw
    default = raw.get("default")
    if has_default and kind in {"vararg", "kwarg"}:
        CompilerExit.user_error(
            f"{source_name}: С„СѓРЅРєС†РёСЏ {fn} arg#{idx}: default РЅРµР»СЊР·СЏ СЃ kind={kind}",
            show_path=False,
            show_pos=False,
        )

    return ApiArgSpec(name=name, ann=ann, kind=kind, default=default, has_default=has_default)


def _parse_decorator(raw: Any, *, fn: str, idx: int, source_name: str) -> ApiDecoratorSpec:
    if isinstance(raw, str):
        return ApiDecoratorSpec(kind="raw", options={"expr": raw})
    if not isinstance(raw, dict):
        CompilerExit.user_error(
            f"{source_name}: {fn} decorator#{idx}: РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ string РёР»Рё mapping",
            show_path=False,
            show_pos=False,
        )
    kind = require_non_empty_str(raw.get("kind"), f"{source_name}: {fn} decorator#{idx} kind").lower()
    options = dict(raw)
    options.pop("kind", None)
    return ApiDecoratorSpec(kind=kind, options=options)


def _apply_class_loc(
    cls: ApiClassSpec,
    *,
    parent_path: str | None,
    lang: str,
    loc_meta: LocMetaSpec,
    loc_classes: dict[str, tuple[Any, str]],
    loc_methods: dict[str, tuple[Any, str]],
    loc_fields: dict[str, tuple[Any, str]],
) -> None:
    cls_path = cls.name if parent_path is None else f"{parent_path}.{cls.name}"

    class_loc = loc_classes.get(cls_path)
    if class_loc is None:
        cls.docstring = None
        cls.deprecated = None
    else:
        payload, source_name = class_loc
        cls.docstring, cls.deprecated = _build_class_doc_payload(payload, source_name, cls_path)

    _apply_members_loc(
        class_path=cls_path,
        lang=lang,
        loc_meta=loc_meta,
        loc_classes=loc_classes,
        loc_methods=loc_methods,
        loc_fields=loc_fields,
        fields=cls.fields,
        methods=cls.methods,
        classes=cls.classes,
    )

    for block in cls.ifs:
        _apply_members_loc(
            class_path=cls_path,
            lang=lang,
            loc_meta=loc_meta,
            loc_classes=loc_classes,
            loc_methods=loc_methods,
            loc_fields=loc_fields,
            fields=block.fields,
            methods=block.methods,
            classes=block.classes,
        )


def _apply_members_loc(
    *,
    class_path: str,
    lang: str,
    loc_meta: LocMetaSpec,
    loc_classes: dict[str, tuple[Any, str]],
    loc_methods: dict[str, tuple[Any, str]],
    loc_fields: dict[str, tuple[Any, str]],
    fields: list[ApiFieldSpec],
    methods: list[ApiFunctionSpec],
    classes: list[ApiClassSpec],
) -> None:
    for field in fields:
        field_loc = loc_fields.get(f"{class_path}.{field.name}")
        if field_loc is None:
            field.docstring = None
        else:
            payload, source_name = field_loc
            field.docstring = _build_field_doc_payload(payload, source_name, f"{class_path}.{field.name}")

    for method in methods:
        method_loc = loc_methods.get(f"{class_path}.{method.name}")
        payload, source_name = _split_loc(method_loc, lang)
        method.docstring, method.deprecated = _build_function_doc_payload(
            entry=payload,
            source_name=source_name,
            fn_name=f"{class_path}.{method.name}",
            args=method.args,
            returns=method.returns,
            loc_meta=loc_meta,
        )

    for nested in classes:
        _apply_class_loc(
            nested,
            parent_path=class_path,
            lang=lang,
            loc_meta=loc_meta,
            loc_classes=loc_classes,
            loc_methods=loc_methods,
            loc_fields=loc_fields,
        )


def _build_function_doc_payload(
    *,
    entry: Any,
    source_name: str,
    fn_name: str,
    args: list[ApiArgSpec],
    returns: str | None,
    loc_meta: LocMetaSpec,
) -> tuple[str | None, str | None]:
    field_name = f"{source_name}: {fn_name}"
    summary: str | None = None
    description: str | None = None
    deprecated_msg: str | None = None
    arg1_text: str | None = None
    arg_desc_map: dict[str, str] = {}
    returns_desc: str | None = None

    if entry is not None:
        if isinstance(entry, str):
            summary = require_non_empty_str(entry, field_name)
        elif isinstance(entry, dict):
            allowed = {"summary", "description", "deprecated", "arg1", "args", "returns"}
            for key in entry:
                if key not in allowed:
                    CompilerExit.user_error(
                        f"{field_name}: РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РєР»СЋС‡ '{key}'",
                        show_path=False,
                        show_pos=False,
                    )
            if entry.get("summary") is not None:
                summary = require_non_empty_str(entry.get("summary"), f"{field_name}.summary")
            if entry.get("description") is not None:
                description = require_non_empty_str(entry.get("description"), f"{field_name}.description")
            if entry.get("deprecated") is not None:
                deprecated_msg = require_non_empty_str(entry.get("deprecated"), f"{field_name}.deprecated")
            if entry.get("arg1") is not None:
                arg1_text = require_non_empty_str(entry.get("arg1"), f"{field_name}.arg1")
            if entry.get("returns") is not None:
                returns_desc = require_non_empty_str(entry.get("returns"), f"{field_name}.returns")

            args_raw = entry.get("args")
            if args_raw is not None:
                if not isinstance(args_raw, dict):
                    CompilerExit.user_error(
                        f"{field_name}.args РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ mapping",
                        show_path=False,
                        show_pos=False,
                    )
                for arg_name_raw, arg_desc_raw in args_raw.items():
                    arg_name = require_identifier(arg_name_raw, f"{field_name}.args.<name>")
                    arg_desc_map[arg_name] = require_non_empty_str(arg_desc_raw, f"{field_name}.args.{arg_name}")
        else:
            CompilerExit.user_error(
                f"{field_name}: РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ string РёР»Рё mapping",
                show_path=False,
                show_pos=False,
            )

    if arg1_text is not None and not args:
        CompilerExit.user_error(
            f"{field_name}.arg1 СѓРєР°Р·Р°РЅ, РЅРѕ Сѓ С„СѓРЅРєС†РёРё РЅРµС‚ Р°СЂРіСѓРјРµРЅС‚РѕРІ",
            show_path=False,
            show_pos=False,
        )
    for arg_name in arg_desc_map:
        if all(arg.name != arg_name for arg in args):
            CompilerExit.user_error(
                f"{field_name}.args.{arg_name}: Р°СЂРіСѓРјРµРЅС‚ РЅРµ РЅР°Р№РґРµРЅ РІ СЃРёРіРЅР°С‚СѓСЂРµ",
                show_path=False,
                show_pos=False,
            )

    sections: list[str] = []
    if summary:
        sections.append(summary)
    if description:
        sections.append(description)

    if args:
        lines = ["Args:"]
        for idx, arg in enumerate(args):
            ann = arg.ann if arg.ann else "Any"
            optional = ", optional" if arg.has_default else ""
            arg_desc = arg_desc_map.get(arg.name)
            if arg_desc is None and idx == 0 and arg1_text is not None:
                arg_desc = arg1_text

            details: list[str] = []
            if arg_desc:
                details.append(arg_desc)
            if arg.has_default:
                details.append(_render_localized_edge_case(loc_meta.defaults_to_template, _render_default_for_doc(arg.default)))

            if details:
                lines.append(f"    {arg.name} ({ann}{optional}): {' '.join(details)}")
            else:
                lines.append(f"    {arg.name} ({ann}{optional})")
        sections.append("\n".join(lines))

    if returns is not None:
        if returns_desc:
            sections.append(f"Returns:\n    {returns}: {returns_desc}")
        else:
            sections.append(f"Returns:\n    {returns}")

    if not sections:
        return None, deprecated_msg

    return "\n\n".join(sections), deprecated_msg


def _build_class_doc_payload(entry: Any, source_name: str, class_path: str) -> tuple[str | None, str | None]:
    if entry is None:
        return None, None
    field = f"{source_name}: classes.{class_path}"
    if isinstance(entry, str):
        return require_non_empty_str(entry, field), None
    if not isinstance(entry, dict):
        CompilerExit.user_error(f"{field}: РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ string РёР»Рё mapping", show_path=False, show_pos=False)

    allowed = {"summary", "description", "deprecated"}
    for key in entry:
        if key not in allowed:
            CompilerExit.user_error(f"{field}: РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РєР»СЋС‡ '{key}'", show_path=False, show_pos=False)

    summary = entry.get("summary")
    description = entry.get("description")
    deprecated = entry.get("deprecated")
    if summary is not None:
        summary = require_non_empty_str(summary, f"{field}.summary")
    if description is not None:
        description = require_non_empty_str(description, f"{field}.description")
    if deprecated is not None:
        deprecated = require_non_empty_str(deprecated, f"{field}.deprecated")

    if summary is None and description is None:
        doc = None
    elif summary is None:
        doc = description
    elif description is None:
        doc = summary
    else:
        doc = f"{summary}\n\n{description}"
    return doc, deprecated


def _build_field_doc_payload(entry: Any, source_name: str, field_path: str) -> str | None:
    if entry is None:
        return None
    field = f"{source_name}: fields.{field_path}"
    if isinstance(entry, str):
        return require_non_empty_str(entry, field)
    if not isinstance(entry, dict):
        CompilerExit.user_error(f"{field}: РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ string РёР»Рё mapping", show_path=False, show_pos=False)

    allowed = {"summary", "description"}
    for key in entry:
        if key not in allowed:
            CompilerExit.user_error(f"{field}: РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РєР»СЋС‡ '{key}'", show_path=False, show_pos=False)
    summary = entry.get("summary")
    description = entry.get("description")
    if summary is not None:
        summary = require_non_empty_str(summary, f"{field}.summary")
    if description is not None:
        description = require_non_empty_str(description, f"{field}.description")
    if summary is None and description is None:
        return None
    if summary is None:
        return description
    if description is None:
        return summary
    return f"{summary}\n\n{description}"


def _validate_loc_keys(mapping: dict[str, tuple[Any, str]], known: set[str], kind: str, lang: str) -> None:
    for key in mapping:
        if key not in known:
            CompilerExit.user_error(
                f"loc/{lang}: РµСЃС‚СЊ Р»РѕРєР°Р»РёР·Р°С†РёСЏ РґР»СЏ РЅРµРёР·РІРµСЃС‚РЅРѕРіРѕ {kind} '{key}'",
                show_path=False,
                show_pos=False,
            )


def _split_loc(loc: tuple[Any, str] | None, lang: str) -> tuple[Any, str]:
    if loc is None:
        return None, f"loc/{lang}: <auto>"
    return loc[0], loc[1]


def _render_default_for_doc(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return repr(value)


def _render_localized_edge_case(template: str, value: str) -> str:
    return template.replace("{value}", value)


def _has_any_deprecated(functions: list[ApiFunctionSpec], classes: list[ApiClassSpec]) -> bool:
    for fn in functions:
        if fn.deprecated:
            return True

    def walk(cls: ApiClassSpec) -> bool:
        if cls.deprecated:
            return True
        for m in cls.methods:
            if m.deprecated:
                return True
        for block in cls.ifs:
            for m in block.methods:
                if m.deprecated:
                    return True
            for nested in block.classes:
                if walk(nested):
                    return True
        return any(walk(n) for n in cls.classes)

    return any(walk(c) for c in classes)


def _collect_yaml_files(root: Path, *, empty_error: str, missing_error: str) -> list[Path]:
    if not root.exists() or not root.is_dir():
        CompilerExit.user_error(missing_error, show_path=False, show_pos=False)

    files = sorted(
        [*root.rglob("*.yml"), *root.rglob("*.yaml")],
        key=lambda p: p.as_posix(),
    )
    if not files:
        CompilerExit.user_error(empty_error, show_path=False, show_pos=False)
    return files


def _load_yaml_file(path: Path, source_name: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as error:
        CompilerExit.internal_error(f"РќРµ СѓРґР°Р»РѕСЃСЊ РїСЂРѕС‡РёС‚Р°С‚СЊ {source_name}: {error}")
    return parse_yaml(text, source_name=source_name)


def _resolve_under(root: Path, rel_path: Path, *, field_name: str) -> Path:
    target = (root / rel_path).resolve()
    root_resolved = root.resolve()
    if root_resolved != target and root_resolved not in target.parents:
        CompilerExit.user_error(
            f"{field_name}: РїСѓС‚СЊ РІС‹С…РѕРґРёС‚ Р·Р° СЂР°Р·СЂРµС€С‘РЅРЅС‹Р№ РєРѕСЂРµРЅСЊ ({rel_path})",
            show_path=False,
            show_pos=False,
        )
    return target


def _ensure_package_inits(source_root: Path, package_dir: Path) -> None:
    current = package_dir
    source_root = source_root.resolve()
    while True:
        if current == source_root:
            break
        init_py = current / "__init__.py"
        if not init_py.exists():
            init_py.write_text("", encoding="utf-8")
        if source_root in current.parents:
            current = current.parent
            continue
        break


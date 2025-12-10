import importlib.util
import sys
import sysconfig
import tokenize
from pathlib import Path
from typing import Sequence

from ....config import Py2GluaConfig
from ...parse import PyLogicNode
from ..ir_dataclass import PyIRContext, PyIRImport, PyIRImportType, PyIRNode

try:
    _py2glua_spec = importlib.util.find_spec("py2glua")
    if _py2glua_spec and getattr(_py2glua_spec, "origin", None):
        PY2GLUA_ROOT = Path(_py2glua_spec.origin).resolve().parent  # pyright: ignore[reportArgumentType]

    else:
        PY2GLUA_ROOT = None

except Exception:
    PY2GLUA_ROOT = None


class ImportBuilder:
    @staticmethod
    def build(
        parent_obj: PyIRNode,
        node: PyLogicNode,
    ) -> Sequence[PyIRNode]:
        origin = node.origins[0]
        if origin is None:
            raise ValueError("Origin is None")

        context: PyIRContext | None = getattr(parent_obj, "context", None)
        if context is None:
            raise ValueError("parent_obj context is missing")

        toks: list[tokenize.TokenInfo] = [
            t for t in origin.tokens if isinstance(t, tokenize.TokenInfo)
        ]
        if not toks:
            return []

        first_non_trivial = next(
            (
                t
                for t in toks
                if t.type
                not in (
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                )
            ),
            None,
        )
        if first_non_trivial is None:
            return []

        line, offset = first_non_trivial.start
        head = first_non_trivial.string

        project_root = Py2GluaConfig.source.resolve()
        std_paths = sysconfig.get_paths()
        std_base = std_paths.get("stdlib") or sys.base_prefix
        stdlib_dir = Path(std_base).resolve()

        cur_file: Path | None = None
        if hasattr(parent_obj, "path") and getattr(parent_obj, "path") is not None:
            cur_file = getattr(parent_obj, "path")

        elif isinstance(context.meta, dict):
            cf = context.meta.get("__file__")
            if isinstance(cf, Path):
                cur_file = cf

        modules: list[str] = []

        if head == "import":
            modules = ImportBuilder._parse_import_stmt(toks, context)

        elif head == "from":
            modules = ImportBuilder._parse_from_stmt(
                toks,
                context=context,
                current_file=cur_file,
                project_root=project_root,
            )

        else:
            return []

        ir_nodes: list[PyIRImport] = []

        for mod in modules:
            i_type = ImportBuilder.resolve_type(mod, project_root, stdlib_dir)
            if i_type == PyIRImportType.EXTERNAL:
                raise RuntimeError(
                    f"External dependency '{mod}' detected. "
                    "External pip packages are not supported in py2glua."
                )

            ir_nodes.append(
                PyIRImport(
                    line=line,
                    offset=offset,
                    modules=[mod],
                    i_type=i_type,
                )
            )
            context.meta["imports"].append(mod)

        return ir_nodes

    @staticmethod
    def _parse_import_stmt(
        toks: list[tokenize.TokenInfo],
        context: PyIRContext,
    ) -> list[str]:
        modules: list[str] = []

        try:
            imp_idx = next(i for i, t in enumerate(toks) if t.string == "import")

        except StopIteration:
            raise SyntaxError("Malformed import statement")

        i = imp_idx + 1
        n = len(toks)

        def skip_trivia(idx: int) -> int:
            while idx < n and toks[idx].type in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            ):
                idx += 1
            return idx

        i = skip_trivia(i)

        while i < n:
            t = toks[i]

            if t.type in (tokenize.NEWLINE, tokenize.NL):
                break

            s = t.string

            if s in {",", ";"}:
                i += 1
                i = skip_trivia(i)
                continue

            if s == "as":
                raise SyntaxError("Malformed import: 'as' without base name")

            if t.type != tokenize.NAME:
                break

            name_parts = [s]
            i += 1

            while (
                i + 1 < n
                and toks[i].type == tokenize.OP
                and toks[i].string == "."
                and toks[i + 1].type == tokenize.NAME
            ):
                name_parts.append(toks[i + 1].string)
                i += 2

            full_name = ".".join(name_parts)
            modules.append(full_name)

            i = skip_trivia(i)
            if i < n and toks[i].string == "as":
                if i + 1 >= n or toks[i + 1].type != tokenize.NAME:
                    raise SyntaxError("Malformed import alias")
                alias_name = toks[i + 1].string
                context.meta["aliases"][alias_name] = full_name
                i += 2
                i = skip_trivia(i)
                continue

            i = skip_trivia(i)

        return modules

    @staticmethod
    def _parse_from_stmt(
        toks: list[tokenize.TokenInfo],
        context: PyIRContext,
        current_file: Path | None,
        project_root: Path,
    ) -> list[str]:
        modules: list[str] = []

        try:
            from_idx = next(i for i, t in enumerate(toks) if t.string == "from")
            imp_idx = next(i for i, t in enumerate(toks) if t.string == "import")

        except StopIteration:
            raise SyntaxError("Malformed 'from' import statement")

        if imp_idx <= from_idx:
            raise SyntaxError("Malformed 'from' import statement")

        base_toks = toks[from_idx + 1 : imp_idx]

        level = 0
        pos = 0
        while (
            pos < len(base_toks)
            and base_toks[pos].type == tokenize.OP
            and base_toks[pos].string == "."
        ):
            level += 1
            pos += 1

        name_parts: list[str] = []
        while pos < len(base_toks):
            bt = base_toks[pos]
            if bt.type == tokenize.NAME:
                name_parts.append(bt.string)
            pos += 1

        module_rel = ".".join(name_parts) if name_parts else None

        if level == 0:
            if module_rel is None:
                raise SyntaxError("Invalid 'from' import without module name")

            base_module = module_rel

        else:
            if current_file is None:
                raise SyntaxError(
                    "Relative import used but current file path is unknown "
                    "(parent_obj.path or context.meta['__file__'] must be set)."
                )

            base_module = ImportBuilder._resolve_relative_module(
                module_rel,
                level,
                current_file,
                project_root,
            )

        name_toks = toks[imp_idx + 1 :]

        if any(nt.string == "*" for nt in name_toks):
            raise SyntaxError("Wildcard import (*) is forbidden in py2glua.")

        i = 0
        n = len(name_toks)

        def skip_trivia(idx: int) -> int:
            while idx < n and name_toks[idx].type in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            ):
                idx += 1
            return idx

        i = skip_trivia(i)

        while i < n:
            t = name_toks[i]

            if t.type in (tokenize.NEWLINE, tokenize.NL):
                break

            s = t.string

            if s in {",", ";"}:
                i += 1
                i = skip_trivia(i)
                continue

            if s == "as":
                raise SyntaxError("Malformed 'from' import: 'as' without name")

            if t.type != tokenize.NAME:
                break

            name = s
            full_name = f"{base_module}.{name}" if base_module else name
            modules.append(full_name)

            i += 1
            i = skip_trivia(i)

            if i < n and name_toks[i].string == "as":
                if i + 1 >= n or name_toks[i + 1].type != tokenize.NAME:
                    raise SyntaxError("Malformed 'from' import alias")
                alias_name = name_toks[i + 1].string
                context.meta["aliases"][alias_name] = full_name
                i += 2
                i = skip_trivia(i)
                continue

        return modules

    @staticmethod
    def _resolve_relative_module(
        module: str | None,
        level: int,
        current_file: Path,
        project_root: Path,
    ) -> str:
        current_file = current_file.resolve()
        project_root = project_root.resolve()

        try:
            rel = current_file.relative_to(project_root)
        except ValueError:
            raise SyntaxError(
                f"Current file {current_file} is not inside project root {project_root}"
            )

        rel = rel.with_suffix("")
        parts = list(rel.parts)
        if parts:
            parts = parts[:-1]

        up = level - 1
        if up > len(parts):
            raise SyntaxError("Relative import goes beyond project root")

        base = parts[: len(parts) - up]

        if module:
            base.extend(module.split("."))

        return ".".join(base)

    @staticmethod
    def resolve_type(
        module: str,
        project_root: Path,
        stdlib_dir: Path,
    ) -> PyIRImportType:
        if not module:
            return PyIRImportType.UNKNOWN

        try:
            spec = importlib.util.find_spec(module)

        except Exception:
            spec = None

        if spec and getattr(spec, "origin", None) in {"built-in", "frozen"}:
            return PyIRImportType.STD_LIB

        if not spec or not getattr(spec, "origin", None):
            if module in sys.builtin_module_names:
                return PyIRImportType.STD_LIB

            return PyIRImportType.UNKNOWN

        try:
            mod_path = Path(spec.origin).resolve()  # pyright: ignore[reportArgumentType]

        except Exception:
            return PyIRImportType.UNKNOWN

        project_root = project_root.resolve()
        stdlib_dir = stdlib_dir.resolve()

        if stdlib_dir in mod_path.parents or mod_path == stdlib_dir:
            return PyIRImportType.STD_LIB

        if PY2GLUA_ROOT is not None:
            if mod_path == PY2GLUA_ROOT or PY2GLUA_ROOT in mod_path.parents:
                return PyIRImportType.INTERNAL

        if project_root in mod_path.parents or mod_path == project_root:
            return PyIRImportType.LOCAL

        paths = sysconfig.get_paths()
        purelib = Path(paths.get("purelib", "") or ".").resolve()
        platlib = Path(paths.get("platlib", "") or ".").resolve()

        if purelib in mod_path.parents or platlib in mod_path.parents:
            return PyIRImportType.EXTERNAL

        return PyIRImportType.UNKNOWN


# TODO: from x import (y, z) can crash all

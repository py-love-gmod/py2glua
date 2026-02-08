from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final

from ....._cli import logger
from ....._cli.ansi import Fore, Style
from ....py.ir_dataclass import (
    PyIRAttribute,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRDecorator,
    PyIRFile,
    PyIRFunctionDef,
    PyIRImport,
    PyIRNode,
)
from ..analysis.symlinks import PyIRSymLink, SymLinkContext
from ..common import (
    canon_path,
    collect_decl_symid_cache,
    decorator_call_parts,
    iter_imported_names,
)


def _warn_deprecated(
    *,
    display_name: str,
    text: str | None,
    path: Path | None,
    node: PyIRNode,
) -> None:
    parts: list[str] = [f"[{Fore.YELLOW}DEPRECATED{Style.RESET_ALL}]"]
    name_colored = f"{Fore.RED}{display_name}{Style.RESET_ALL}"
    parts.append(f"Используется устаревший API: {name_colored}")

    if text:
        parts.append(text.replace("\\n", "\n").strip())

    if path is None:
        parts.append("Файл: неизвестно")

    else:
        parts.append(f"Файл: {path.resolve()}")

    parts.append(f"LINE|OFFSET: {node.line or 0}|{node.offset or 0}")

    logger.warning("\n".join(parts))


class CollectDeprecatedDeclsPass:
    """
    Analysis-pass (phase 1):
    Collects deprecated definitions and their messages from @warnings.deprecated(...).

    Supports:
      - `from warnings import deprecated`
      - `from warnings import deprecated as dep`
      - `import warnings` + `@warnings.deprecated(...)`

    Writes into ctx:
      - ctx._deprecated_symid_to_msg: dict[int, str]
      - ctx._deprecated_symid_to_disp: dict[int, str]     (pretty name for warnings)
      - ctx._deprecated_method_to_symid: dict[(int, str), int]  ((class_symid, method_name) -> method_symid)
      - ctx._deprecated_method_to_msg: dict[(int, str), str]
      - ctx._deprecated_classname_by_symid: dict[int, str]
      - ctx._deprecated_symid_by_decl_cache: dict[(Path, line, name) -> symid]
    """

    _SYM_MSG: Final[str] = "_deprecated_symid_to_msg"
    _SYM_DISP: Final[str] = "_deprecated_symid_to_disp"
    _METH_SYMID: Final[str] = "_deprecated_method_to_symid"
    _METH_MSG: Final[str] = "_deprecated_method_to_msg"
    _CLSNAME: Final[str] = "_deprecated_classname_by_symid"

    _DECL_CACHE: Final[str] = "_deprecated_symid_by_decl_cache"

    _WARNINGS_MOD: Final[str] = "warnings"
    _DEPR_NAME: Final[str] = "deprecated"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("CollectDeprecatedDeclsPass.run ожидает PyIRFile")

        if ir.path is None:
            return ir

        sym_msg = ctx._deprecated_symid_to_msg
        sym_disp = ctx._deprecated_symid_to_disp
        meth_symid = ctx._deprecated_method_to_symid
        meth_msg = ctx._deprecated_method_to_msg
        clsname = ctx._deprecated_classname_by_symid

        symid_by_decl = collect_decl_symid_cache(ctx, cache_field=cls._DECL_CACHE)
        fp = canon_path(ir.path)
        if fp is None:
            return ir

        deprecated_bindings, warnings_module_bindings = (
            cls._collect_deprecated_bindings(ir)
        )

        if not deprecated_bindings and not warnings_module_bindings:
            return ir

        class_stack: list[int] = []

        def node_symid_for_def(n: PyIRNode) -> int | None:
            if not isinstance(n, (PyIRFunctionDef, PyIRClassDef)):
                return None
            if n.line is None:
                return None
            return symid_by_decl.get((fp, n.line, n.name))

        def walk_body(body: list[PyIRNode]) -> None:
            for st in body:
                if isinstance(st, PyIRClassDef):
                    sid = node_symid_for_def(st)
                    if sid is not None:
                        class_stack.append(int(sid))
                        clsname[int(sid)] = st.name

                    walk_body(st.body)

                    if sid is not None:
                        class_stack.pop()
                    continue

                if isinstance(st, PyIRFunctionDef):
                    msg = cls._extract_deprecated_message(
                        st.decorators,
                        deprecated_bindings=deprecated_bindings,
                        warnings_module_bindings=warnings_module_bindings,
                    )
                    if msg is not None:
                        sid = node_symid_for_def(st)
                        if sid is not None:
                            sid_i = int(sid)
                            sym_msg[sid_i] = msg

                            if class_stack:
                                csid = int(class_stack[-1])
                                cname = clsname.get(csid, "<class>")
                                sym_disp[sid_i] = f"{cname}.{st.name}"
                                meth_symid[(csid, st.name)] = sid_i
                                meth_msg[(csid, st.name)] = msg
                            else:
                                sym_disp[sid_i] = st.name

                    walk_body(st.body)
                    continue

                if is_dataclass(st):
                    for f in fields(st):
                        v = getattr(st, f.name)
                        if isinstance(v, list):
                            nested = [x for x in v if isinstance(x, PyIRNode)]
                            if nested:
                                walk_body(nested)

        walk_body(ir.body)
        return ir

    @classmethod
    def _collect_deprecated_bindings(cls, ir: PyIRFile) -> tuple[set[str], set[str]]:
        """
        Returns:
          - deprecated_bindings: local names bound to warnings.deprecated
          - warnings_module_bindings: local names bound to warnings module (for warnings.deprecated form)
        """
        deprecated_bindings: set[str] = set()
        warnings_module_bindings: set[str] = set()

        for n in ir.walk():
            if not isinstance(n, PyIRImport):
                continue

            modules = getattr(n, "modules", None) or []
            mod = modules[0] if isinstance(modules, list) and modules else ""
            names = getattr(n, "names", None)

            if n.if_from and mod == cls._WARNINGS_MOD and names:
                for orig, bound in iter_imported_names(names):
                    if orig == cls._DEPR_NAME:
                        deprecated_bindings.add(bound)

            if (not n.if_from) and mod == cls._WARNINGS_MOD:
                if names:
                    for orig, bound in iter_imported_names(names):
                        if orig == cls._WARNINGS_MOD:
                            warnings_module_bindings.add(bound)
                else:
                    warnings_module_bindings.add(cls._WARNINGS_MOD)

        return deprecated_bindings, warnings_module_bindings

    @classmethod
    def _extract_deprecated_message(
        cls,
        decorators: list[PyIRDecorator],
        *,
        deprecated_bindings: set[str],
        warnings_module_bindings: set[str],
    ) -> str | None:
        """
        Returns message string if a decorator is warnings.deprecated(...).
        Requires a proven binding in this file.
        """
        for d in decorators or []:
            fn_expr, args_p, args_kw = decorator_call_parts(d)

            if not cls._is_deprecated_fn_expr(
                fn_expr,
                deprecated_bindings=deprecated_bindings,
                warnings_module_bindings=warnings_module_bindings,
            ):
                continue

            msg = cls._pick_message(args_p, args_kw)

            return msg if msg is not None else ""

        return None

    @classmethod
    def _is_deprecated_fn_expr(
        cls,
        expr: PyIRNode,
        *,
        deprecated_bindings: set[str],
        warnings_module_bindings: set[str],
    ) -> bool:

        if type(expr).__name__ == "PyIRVarUse":
            name = getattr(expr, "name", None)
            return bool(name) and name in deprecated_bindings

        if type(expr).__name__ == "PyIRSymLink":
            name = getattr(expr, "name", None)
            return bool(name) and name in deprecated_bindings

        if isinstance(expr, PyIRAttribute) and expr.attr == cls._DEPR_NAME:
            base = expr.value
            if type(base).__name__ == "PyIRVarUse":
                bname = getattr(base, "name", None)
                return bool(bname) and bname in warnings_module_bindings
            if type(base).__name__ == "PyIRSymLink":
                bname = getattr(base, "name", None)
                return bool(bname) and bname in warnings_module_bindings

        return False

    @staticmethod
    def _pick_message(
        args_p: list[PyIRNode], args_kw: dict[str, PyIRNode]
    ) -> str | None:

        if args_p:
            a0 = args_p[0]
            if isinstance(a0, PyIRConstant) and isinstance(a0.value, str):
                return a0.value

        for key in ("message", "msg", "reason", "text"):
            v = args_kw.get(key)
            if isinstance(v, PyIRConstant) and isinstance(v.value, str):
                return v.value

        for v in args_kw.values():
            if isinstance(v, PyIRConstant) and isinstance(v.value, str):
                return v.value

        return None


class WarnDeprecatedUsesPass:
    """
    Analysis-pass (phase 2):
    Emits warnings when deprecated APIs are called.

    Uses ctx data from CollectDeprecatedDeclsPass.

    Warn patterns:
      - direct call:   foo(...) where foo resolves to deprecated symid
      - class method:  Class.method(...) where Class resolves to a class_symid
                       and (class_symid, "method") is deprecated

    Prints:
      - file + line/offset of the CALL SITE
      - message from @deprecated("...")

    Deduplicates by (file, line, offset, display_name).
    """

    _SYM_MSG: Final[str] = "_deprecated_symid_to_msg"
    _SYM_DISP: Final[str] = "_deprecated_symid_to_disp"
    _METH_MSG: Final[str] = "_deprecated_method_to_msg"
    _CLSNAME: Final[str] = "_deprecated_classname_by_symid"

    _SEEN: Final[str] = "_deprecated_warned_callsites"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        if not isinstance(ir, PyIRFile):
            raise TypeError("WarnDeprecatedUsesPass.run ожидает PyIRFile")

        if ir.path is None:
            return ir

        sym_msg = ctx._deprecated_symid_to_msg
        meth_msg = ctx._deprecated_method_to_msg
        sym_disp = ctx._deprecated_symid_to_disp
        clsname = ctx._deprecated_classname_by_symid

        if not sym_msg and not meth_msg:
            return ir

        seen = ctx._deprecated_warned_callsites

        fp = canon_path(ir.path) or ir.path

        def warn(display_name: str, msg: str, call: PyIRCall) -> None:
            line = call.line
            off = call.offset
            key = (str(fp), int(line or -1), int(off or -1), display_name)
            if key in seen:
                return
            seen.add(key)

            text = msg.strip()
            _warn_deprecated(
                display_name=display_name,
                text=text,
                path=ir.path,
                node=call,
            )

        def rw(node: PyIRNode) -> None:
            if isinstance(node, PyIRCall):
                if isinstance(node.func, PyIRSymLink):
                    sid = int(node.func.symbol_id)
                    msg = sym_msg.get(sid)
                    if isinstance(msg, str):
                        disp = sym_disp.get(sid) or node.func.name
                        warn(str(disp), msg, node)

                if isinstance(node.func, PyIRAttribute) and isinstance(
                    node.func.value, PyIRSymLink
                ):
                    csid = int(node.func.value.symbol_id)
                    mname = node.func.attr
                    msg = meth_msg.get((csid, mname))
                    if isinstance(msg, str):
                        cname = clsname.get(csid, node.func.value.name)
                        disp = f"{cname}.{mname}"
                        warn(disp, msg, node)

                rw(node.func)
                for a in node.args_p:
                    rw(a)

                for a in node.args_kw.values():
                    rw(a)

                return

            if isinstance(node, PyIRAttribute):
                rw(node.value)
                return

            if is_dataclass(node):
                for f in fields(node):
                    v = getattr(node, f.name)
                    if isinstance(v, PyIRNode):
                        rw(v)
                    elif isinstance(v, list):
                        for x in v:
                            if isinstance(x, PyIRNode):
                                rw(x)
                    elif isinstance(v, dict):
                        for x in v.values():
                            if isinstance(x, PyIRNode):
                                rw(x)

        for st in ir.body:
            rw(st)

        return ir

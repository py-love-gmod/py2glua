from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Set, Tuple

from ....._cli import logger
from ....py.ir_dataclass import (
    LuaNil,
    PyBinOPType,
    PyIRAssign,
    PyIRAttribute,
    PyIRBinOP,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRNode,
    PyIRReturn,
    PyIRUnaryOP,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
    PyUnaryOPType,
)
from .symlinks import PyIRSymLink, SymLinkContext

TYPE_NONE = "None"
TYPE_NIL = "Nil"
TYPE_VALUE = "Value"


def _freeze_env(env: Dict[int, Set[str]]) -> Dict[int, frozenset[str]]:
    return {k: frozenset(v) for k, v in env.items()}


class TypeFlowPass:
    """
    Type analysis from scratch.

    Model:
      - None, Nil, Value are 3 distinct atoms: None != Nil != Value
      - Other names are treated as nominal types: "Player", "Entity", "NPC", ...

    Stores in ctx (dynamic fields):
      - ctx.type_sets_by_symbol_id: dict[int, frozenset[str]]
      - ctx.type_env_by_uid: dict[int, dict[int, frozenset[str]]]
      - ctx._typeflow_warned: set[str]

    Extra registries in ctx (for inference):
      - ctx.fn_return_types_by_symbol_id: dict[int, frozenset[str]]
      - ctx.method_return_types_by_class: dict[tuple[str, str], frozenset[str]]
      - ctx.fn_return_ann_by_symbol_id: dict[int, str|None]
      - ctx.method_return_ann_by_class: dict[tuple[str, str], str|None]
      - ctx.class_bases_by_name: dict[str, tuple[str, ...]]

    Key feature:
      - Infers local types from RETURN TYPES of functions/methods:
          x = foo()    -> x gets foo.returns
          x = obj.m()  -> x gets m.returns resolved by receiver type (+bases)

    Narrowing:
      - if x is/== None / x is not/!= None
      - if x is/== nil-like / x is not/!= nil-like
        where nil-like means: nil name, nil(), t.nil, LuaNil constant
      - if IsValid(x): true branch removes None and Nil from x type-set
      - mismatch None vs Nil checks => WARN + narrow best-effort

    Loops:
      - while <cond>: body sees env_true (narrowed) but after loop returns original env
      - for target in iter: body sees target type inferred from iter return annotation if possible,
        but after loop returns original env (no guarantees)
    """

    TYPES_MODULE = "py2glua.glua.core.types"
    _GUARD_NAMES = {"IsValid"}

    _ITERABLE_PREFIXES = (
        "list",
        "set",
        "Sequence",
        "Iterable",
        "Iterator",
        "List",
        "Set",
    )

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        type_sets, snapshots, warn_once = TypeFlowPass._ensure_storage(ctx)
        nil_ids, types_alias_ids = TypeFlowPass._collect_nil_ids(ir, ctx)

        (
            fn_ret_by_sym,
            method_ret_by_class,
            fn_ret_ann_by_sym,
            method_ret_ann_by_class,
            class_bases_by_name,
        ) = TypeFlowPass._collect_return_types(ir, ctx, warn_once)

        for node in ir.walk():
            if isinstance(node, PyIRFunctionDef):
                TypeFlowPass._process_fn(
                    ir,
                    ctx,
                    node,
                    type_sets,
                    snapshots,
                    warn_once,
                    nil_ids,
                    types_alias_ids,
                    fn_ret_by_sym,
                    method_ret_by_class,
                    fn_ret_ann_by_sym,
                    method_ret_ann_by_class,
                    class_bases_by_name,
                )

    @staticmethod
    def _ensure_storage(
        ctx: SymLinkContext,
    ) -> tuple[
        Dict[int, frozenset[str]],
        Dict[int, Dict[int, frozenset[str]]],
        Callable,
    ]:
        type_sets = getattr(ctx, "type_sets_by_symbol_id", None)
        if not isinstance(type_sets, dict):
            type_sets = {}

        snapshots = getattr(ctx, "type_env_by_uid", None)
        if not isinstance(snapshots, dict):
            snapshots = {}

        warned = getattr(ctx, "_typeflow_warned", None)
        if not isinstance(warned, set):
            warned = set()

        setattr(ctx, "type_sets_by_symbol_id", type_sets)
        setattr(ctx, "type_env_by_uid", snapshots)
        setattr(ctx, "_typeflow_warned", warned)

        def warn_once(key: str, msg: str) -> None:
            if key in warned:
                return
            warned.add(key)
            logger.warning(msg)

        return (
            type_sets,  # type: ignore[return-value]
            snapshots,  # type: ignore[return-value]
            warn_once,
        )

    @staticmethod
    def _collect_nil_ids(
        ir: PyIRFile, ctx: SymLinkContext
    ) -> tuple[set[int], set[int]]:
        """
        Collect symbol_id for:
          - imported `nil` (from types import nil as alias)
          - canonical types.nil export
          - module alias for `import types as t` (t.nil)
        """
        try:
            ctx.ensure_module_index()
        except Exception:
            return set(), set()

        if ir.path is None:
            return set(), set()

        current_module = ctx.module_name_by_path.get(ir.path)
        if not current_module:
            return set(), set()

        def iter_imported_names(names):
            for n in names:
                if isinstance(n, tuple):
                    yield n[0], n[1]
                else:
                    yield n, n

        def local_symbol_id(name: str) -> int | None:
            sym = ctx.get_exported_symbol(current_module, name)
            return sym.id.value if sym is not None else None

        nil_canon_sym = ctx.get_exported_symbol(TypeFlowPass.TYPES_MODULE, "nil")
        nil_canon_id = nil_canon_sym.id.value if nil_canon_sym is not None else None

        nil_name_ids: set[int] = set()
        types_module_alias_ids: set[int] = set()

        for node in ir.walk():
            if not isinstance(node, PyIRImport):
                continue

            mod = ".".join(node.modules) if node.modules else ""
            if mod != TypeFlowPass.TYPES_MODULE:
                continue

            if node.if_from:
                for orig, bound in iter_imported_names(node.names):
                    if orig != "nil":
                        continue
                    sid = local_symbol_id(bound)
                    if sid is not None:
                        nil_name_ids.add(sid)
                continue

            if node.names:
                for _orig, bound in iter_imported_names(node.names):
                    sid = local_symbol_id(bound)
                    if sid is not None:
                        types_module_alias_ids.add(sid)

        nil_ids: set[int] = set(nil_name_ids)
        if nil_canon_id is not None:
            nil_ids.add(nil_canon_id)

        return nil_ids, types_module_alias_ids

    @staticmethod
    def _collect_return_types(
        ir: PyIRFile,
        ctx: SymLinkContext,
        warn_once,
    ) -> tuple[
        dict[int, frozenset[str]],
        dict[tuple[str, str], frozenset[str]],
        dict[int, str | None],
        dict[tuple[str, str], str | None],
        dict[str, tuple[str, ...]],
    ]:
        """
        Build local registries for this file and merge into ctx globals:

          - fn_ret_by_sym: symbol_id(func) -> return atoms
          - method_ret_by_class: (ClassName, method_name) -> return atoms

          - fn_ret_ann_by_sym: symbol_id(func) -> raw returns string
          - method_ret_ann_by_class: (ClassName, method_name) -> raw returns string

          - class_bases_by_name: ClassName -> tuple[BaseName,...] (best-effort, file-local)
        """

        fn_ret_by_sym: dict[int, frozenset[str]] = {}
        method_ret_by_class: dict[tuple[str, str], frozenset[str]] = {}

        fn_ret_ann_by_sym: dict[int, str | None] = {}
        method_ret_ann_by_class: dict[tuple[str, str], str | None] = {}

        class_bases_by_name: dict[str, tuple[str, ...]] = {}

        file_uid = ctx.uid(ir)
        file_scope = ctx.file_scope.get(file_uid)
        if file_scope is None:
            return (
                fn_ret_by_sym,
                method_ret_by_class,
                fn_ret_ann_by_sym,
                method_ret_ann_by_class,
                class_bases_by_name,
            )

        def resolve_in_file(name: str) -> int | None:
            link = ctx.resolve(scope=file_scope, name=name)
            return link.target.value if link is not None else None

        def base_name(n: PyIRNode) -> str | None:
            if isinstance(n, PyIRVarUse):
                return n.name

            if isinstance(n, PyIRSymLink):
                return n.name

            return None

        for st in ir.body:
            if isinstance(st, PyIRFunctionDef):
                sid = resolve_in_file(st.name)
                if sid is None:
                    continue

                fn_ret_ann_by_sym[sid] = st.returns
                fn_ret_by_sym[sid] = frozenset(
                    TypeFlowPass._parse_annotation_atoms(st.returns)
                )
                continue

            if isinstance(st, PyIRClassDef):
                cls_name = st.name

                bases: list[str] = []
                for b in st.bases:
                    bn = base_name(b)
                    if bn is not None:
                        bases.append(bn)
                class_bases_by_name[cls_name] = tuple(bases)

                for m in st.body:
                    if not isinstance(m, PyIRFunctionDef):
                        continue

                    method_ret_ann_by_class[(cls_name, m.name)] = m.returns
                    method_ret_by_class[(cls_name, m.name)] = frozenset(
                        TypeFlowPass._parse_annotation_atoms(m.returns)
                    )

        global_fn_ret = getattr(ctx, "fn_return_types_by_symbol_id", None)
        if not isinstance(global_fn_ret, dict):
            global_fn_ret = {}

        global_method_ret = getattr(ctx, "method_return_types_by_class", None)
        if not isinstance(global_method_ret, dict):
            global_method_ret = {}

        global_fn_ret_ann = getattr(ctx, "fn_return_ann_by_symbol_id", None)
        if not isinstance(global_fn_ret_ann, dict):
            global_fn_ret_ann = {}

        global_method_ret_ann = getattr(ctx, "method_return_ann_by_class", None)
        if not isinstance(global_method_ret_ann, dict):
            global_method_ret_ann = {}

        global_class_bases = getattr(ctx, "class_bases_by_name", None)
        if not isinstance(global_class_bases, dict):
            global_class_bases = {}

        for k, v in fn_ret_by_sym.items():
            global_fn_ret[k] = v
        for k, v in method_ret_by_class.items():
            global_method_ret[k] = v

        for k, v in fn_ret_ann_by_sym.items():
            global_fn_ret_ann[k] = v
        for k, v in method_ret_ann_by_class.items():
            global_method_ret_ann[k] = v

        for k, v in class_bases_by_name.items():
            global_class_bases.setdefault(k, v)

        setattr(ctx, "fn_return_types_by_symbol_id", global_fn_ret)
        setattr(ctx, "method_return_types_by_class", global_method_ret)
        setattr(ctx, "fn_return_ann_by_symbol_id", global_fn_ret_ann)
        setattr(ctx, "method_return_ann_by_class", global_method_ret_ann)
        setattr(ctx, "class_bases_by_name", global_class_bases)

        return (
            fn_ret_by_sym,
            method_ret_by_class,
            fn_ret_ann_by_sym,
            method_ret_ann_by_class,
            class_bases_by_name,
        )

    @staticmethod
    def _strip_ws(s: str) -> str:
        return "".join(ch for ch in s if ch not in (" ", "\t", "\n", "\r"))

    @staticmethod
    def _parse_annotation_atoms(ann: str | None) -> Set[str]:
        """
        Minimal atom parser:
          - supports union with '|'
          - supports direct names: None, nil, Value, Player, Entity...
          - if contains [](){} => {Value} (we treat complex types as Value here)
        """
        if not ann:
            return {TYPE_VALUE}

        s = TypeFlowPass._strip_ws(ann)
        if not s:
            return {TYPE_VALUE}

        if any(ch in s for ch in "[](){}"):
            return {TYPE_VALUE}

        parts = [p for p in s.split("|") if p]
        out: Set[str] = set()

        for p in parts:
            if "." in p:
                p = p.split(".")[-1]

            if p in ("None", "NoneType"):
                out.add(TYPE_NONE)

            elif p in ("nil", "LuaNil", "LuaNilType"):
                out.add(TYPE_NIL)

            elif p == "Value":
                out.add(TYPE_VALUE)

            else:
                out.add(p)

        if not out:
            out.add(TYPE_VALUE)

        return out

    @staticmethod
    def _extract_single_generic_arg(s: str, prefix: str) -> str | None:
        """
        Accepts Base[T] with single top-level [] and returns T.
        Best-effort only (no nested generics).
        """
        if not (s.startswith(prefix + "[") and s.endswith("]")):
            return None

        if s.count("[") != 1 or s.count("]") != 1:
            return None

        inner = s[len(prefix) + 1 : -1]
        return inner or None

    @staticmethod
    def _parse_iterable_elem_atoms_from_ann(ann: str | None) -> Set[str] | None:
        """
        Extract element type atoms from list[T]/set[T]/Iterable[T]/Sequence[T] etc.
        Returns None if not recognized.
        """
        if not ann:
            return None

        s = TypeFlowPass._strip_ws(ann)

        lb = s.find("[")
        if lb == -1:
            return None

        base = s[:lb]
        if "." in base:
            base = base.split(".")[-1]

        s_norm = base + s[lb:]

        for prefix in TypeFlowPass._ITERABLE_PREFIXES:
            inner = TypeFlowPass._extract_single_generic_arg(s_norm, prefix)
            if inner is None:
                continue

            return TypeFlowPass._parse_annotation_atoms(inner)

        return None

    @staticmethod
    def _process_fn(
        ir: PyIRFile,
        ctx: SymLinkContext,
        fn: PyIRFunctionDef,
        type_sets: Dict[int, frozenset[str]],
        snapshots: Dict[int, Dict[int, frozenset[str]]],
        warn_once,
        nil_ids: set[int],
        types_alias_ids: set[int],
        fn_ret_by_sym: dict[int, frozenset[str]],
        method_ret_by_class: dict[tuple[str, str], frozenset[str]],
        fn_ret_ann_by_sym: dict[int, str | None],
        method_ret_ann_by_class: dict[tuple[str, str], str | None],
        class_bases_by_name: dict[str, tuple[str, ...]],
    ) -> None:
        fn_uid = ctx.uid(fn)
        body_scope = ctx.fn_body_scope.get(fn_uid)
        if body_scope is None:
            return

        env: Dict[int, Set[str]] = {}

        for param, (ann, _default) in fn.signature.items():
            link = ctx.resolve(scope=body_scope, name=param)
            if link is None:
                continue

            sid = link.target.value
            atoms = TypeFlowPass._parse_annotation_atoms(ann)
            env[sid] = set(atoms)
            type_sets[sid] = frozenset(atoms)

        TypeFlowPass._walk_stmt_list(
            ir,
            ctx,
            fn.body,
            env,
            type_sets,
            snapshots,
            warn_once,
            nil_ids,
            types_alias_ids,
            fn_ret_by_sym,
            method_ret_by_class,
            fn_ret_ann_by_sym,
            method_ret_ann_by_class,
            class_bases_by_name,
        )

    @staticmethod
    def _walk_stmt_list(
        ir: PyIRFile,
        ctx: SymLinkContext,
        stmts: List[PyIRNode],
        env: Dict[int, Set[str]],
        type_sets: Dict[int, frozenset[str]],
        snapshots: Dict[int, Dict[int, frozenset[str]]],
        warn_once,
        nil_ids: set[int],
        types_alias_ids: set[int],
        fn_ret_by_sym: dict[int, frozenset[str]],
        method_ret_by_class: dict[tuple[str, str], frozenset[str]],
        fn_ret_ann_by_sym: dict[int, str | None],
        method_ret_ann_by_class: dict[tuple[str, str], str | None],
        class_bases_by_name: dict[str, tuple[str, ...]],
    ) -> Tuple[Dict[int, Set[str]], bool]:
        cur_env = env

        for st in stmts:
            snapshots[ctx.uid(st)] = _freeze_env(cur_env)

            cur_env, returned = TypeFlowPass._walk_stmt(
                ir,
                ctx,
                st,
                cur_env,
                type_sets,
                snapshots,
                warn_once,
                nil_ids,
                types_alias_ids,
                fn_ret_by_sym,
                method_ret_by_class,
                fn_ret_ann_by_sym,
                method_ret_ann_by_class,
                class_bases_by_name,
            )
            if returned:
                return cur_env, True

        return cur_env, False

    @staticmethod
    def _walk_stmt(
        ir: PyIRFile,
        ctx: SymLinkContext,
        st: PyIRNode,
        env: Dict[int, Set[str]],
        type_sets: Dict[int, frozenset[str]],
        snapshots: Dict[int, Dict[int, frozenset[str]]],
        warn_once,
        nil_ids: set[int],
        types_alias_ids: set[int],
        fn_ret_by_sym: dict[int, frozenset[str]],
        method_ret_by_class: dict[tuple[str, str], frozenset[str]],
        fn_ret_ann_by_sym: dict[int, str | None],
        method_ret_ann_by_class: dict[tuple[str, str], str | None],
        class_bases_by_name: dict[str, tuple[str, ...]],
    ) -> Tuple[Dict[int, Set[str]], bool]:
        if isinstance(st, PyIRReturn):
            TypeFlowPass._walk_expr(ctx, st.value, env, snapshots)
            return env, True

        if (
            isinstance(st, PyIRAssign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], PyIRSymLink)
        ):
            t = st.targets[0]
            v = st.value

            TypeFlowPass._walk_expr(ctx, v, env, snapshots)

            new_atoms = TypeFlowPass._infer_expr_atoms(
                ctx,
                v,
                env,
                nil_ids,
                types_alias_ids,
                fn_ret_by_sym,
                method_ret_by_class,
                warn_once,
                fn_ret_ann_by_sym,
                method_ret_ann_by_class,
                class_bases_by_name,
            )

            env2 = dict(env)
            env2[t.symbol_id] = set(new_atoms)
            type_sets[t.symbol_id] = frozenset(new_atoms)
            return env2, False

        if isinstance(st, PyIRIf):
            TypeFlowPass._walk_expr(ctx, st.test, env, snapshots)

            env_true, env_false = TypeFlowPass._narrow_from_test(
                ctx, st.test, env, warn_once, nil_ids, types_alias_ids
            )

            env_then, ret_then = TypeFlowPass._walk_stmt_list(
                ir,
                ctx,
                st.body,
                env_true,
                type_sets,
                snapshots,
                warn_once,
                nil_ids,
                types_alias_ids,
                fn_ret_by_sym,
                method_ret_by_class,
                fn_ret_ann_by_sym,
                method_ret_ann_by_class,
                class_bases_by_name,
            )
            env_else, ret_else = TypeFlowPass._walk_stmt_list(
                ir,
                ctx,
                st.orelse,
                env_false,
                type_sets,
                snapshots,
                warn_once,
                nil_ids,
                types_alias_ids,
                fn_ret_by_sym,
                method_ret_by_class,
                fn_ret_ann_by_sym,
                method_ret_ann_by_class,
                class_bases_by_name,
            )

            if ret_then and not ret_else:
                return env_else, False

            if ret_else and not ret_then:
                return env_then, False

            if ret_then and ret_else:
                return env, True

            return TypeFlowPass._merge_env(env_then, env_else), False

        if isinstance(st, PyIRWhile):
            TypeFlowPass._walk_expr(ctx, st.test, env, snapshots)

            env_true, _env_false = TypeFlowPass._narrow_from_test(
                ctx,
                st.test,
                env,
                warn_once,
                nil_ids,
                types_alias_ids,
            )

            _env_body, _ = TypeFlowPass._walk_stmt_list(
                ir,
                ctx,
                st.body,
                env_true,
                type_sets,
                snapshots,
                warn_once,
                nil_ids,
                types_alias_ids,
                fn_ret_by_sym,
                method_ret_by_class,
                fn_ret_ann_by_sym,
                method_ret_ann_by_class,
                class_bases_by_name,
            )

            return env, False

        if isinstance(st, PyIRFor):
            TypeFlowPass._walk_expr(ctx, st.iter, env, snapshots)

            env_body = dict(env)

            if isinstance(st.target, PyIRSymLink):
                elem_atoms: Set[str] | None = None

                if isinstance(st.iter, PyIRCall):
                    ann_list = TypeFlowPass._infer_call_return_ann_list(
                        ctx,
                        st.iter,
                        env,
                        fn_ret_ann_by_sym,
                        method_ret_ann_by_class,
                        class_bases_by_name,
                    )
                    merged: Set[str] = set()
                    for ann in ann_list:
                        a = TypeFlowPass._parse_iterable_elem_atoms_from_ann(ann)
                        if a is not None:
                            merged |= set(a)
                    if merged:
                        elem_atoms = merged

                if elem_atoms is None:
                    elem_atoms = {TYPE_VALUE}

                env_body[st.target.symbol_id] = set(elem_atoms)

            else:
                TypeFlowPass._walk_expr(ctx, st.target, env, snapshots)

            _env_body, _ = TypeFlowPass._walk_stmt_list(
                ir,
                ctx,
                st.body,
                env_body,
                type_sets,
                snapshots,
                warn_once,
                nil_ids,
                types_alias_ids,
                fn_ret_by_sym,
                method_ret_by_class,
                fn_ret_ann_by_sym,
                method_ret_ann_by_class,
                class_bases_by_name,
            )

            return env, False

        if isinstance(st, PyIRWithItem):
            TypeFlowPass._walk_expr(ctx, st.context_expr, env, snapshots)
            if st.optional_vars is not None:
                TypeFlowPass._walk_expr(ctx, st.optional_vars, env, snapshots)
            return env, False

        if isinstance(st, PyIRWith):
            for it in st.items:
                TypeFlowPass._walk_expr(ctx, it.context_expr, env, snapshots)
                if it.optional_vars is not None:
                    TypeFlowPass._walk_expr(ctx, it.optional_vars, env, snapshots)

            _env_body, _ = TypeFlowPass._walk_stmt_list(
                ir,
                ctx,
                st.body,
                env,
                type_sets,
                snapshots,
                warn_once,
                nil_ids,
                types_alias_ids,
                fn_ret_by_sym,
                method_ret_by_class,
                fn_ret_ann_by_sym,
                method_ret_ann_by_class,
                class_bases_by_name,
            )
            return env, False

        TypeFlowPass._walk_expr(ctx, st, env, snapshots)
        return env, False

    @staticmethod
    def _walk_expr(
        ctx: SymLinkContext,
        expr: PyIRNode,
        env: Dict[int, Set[str]],
        snapshots: Dict[int, Dict[int, frozenset[str]]],
    ) -> None:
        if isinstance(expr, PyIRCall):
            snapshots[ctx.uid(expr)] = _freeze_env(env)

            TypeFlowPass._walk_expr(ctx, expr.func, env, snapshots)
            for a in expr.args_p:
                TypeFlowPass._walk_expr(ctx, a, env, snapshots)

            for a in expr.args_kw.values():
                TypeFlowPass._walk_expr(ctx, a, env, snapshots)

            return

        if isinstance(expr, PyIRUnaryOP):
            TypeFlowPass._walk_expr(ctx, expr.value, env, snapshots)
            return

        if isinstance(expr, PyIRBinOP):
            TypeFlowPass._walk_expr(ctx, expr.left, env, snapshots)
            TypeFlowPass._walk_expr(ctx, expr.right, env, snapshots)
            return

        if isinstance(expr, PyIRAttribute):
            TypeFlowPass._walk_expr(ctx, expr.value, env, snapshots)
            return

    @staticmethod
    def _infer_expr_atoms(
        ctx: SymLinkContext,
        expr: PyIRNode,
        env: Dict[int, Set[str]],
        nil_ids: set[int],
        types_alias_ids: set[int],
        fn_ret_by_sym: dict[int, frozenset[str]],
        method_ret_by_class: dict[tuple[str, str], frozenset[str]],
        warn_once,
        fn_ret_ann_by_sym: dict[int, str | None],
        method_ret_ann_by_class: dict[tuple[str, str], str | None],
        class_bases_by_name: dict[str, tuple[str, ...]],
    ) -> Set[str]:
        if isinstance(expr, PyIRSymLink):
            return set(env.get(expr.symbol_id, {TYPE_VALUE}))

        if isinstance(expr, PyIRConstant) and expr.value is None:
            return {TYPE_NONE}

        if isinstance(expr, PyIRConstant):
            try:
                if expr.value is LuaNil or isinstance(expr.value, LuaNil):  # type: ignore[arg-type]
                    return {TYPE_NIL}
            except TypeError:
                if expr.value is LuaNil:
                    return {TYPE_NIL}

        if TypeFlowPass._rhs_is_nil(expr, nil_ids, types_alias_ids):
            return {TYPE_NIL}

        if isinstance(expr, PyIRCall):
            return TypeFlowPass._infer_call_return_atoms(
                ctx,
                expr,
                env,
                fn_ret_by_sym,
                method_ret_by_class,
                warn_once,
                class_bases_by_name,
            )

        return {TYPE_VALUE}

    @staticmethod
    def _infer_call_return_atoms(
        ctx: SymLinkContext,
        call: PyIRCall,
        env: Dict[int, Set[str]],
        fn_ret_by_sym: dict[int, frozenset[str]],
        method_ret_by_class: dict[tuple[str, str], frozenset[str]],
        warn_once,
        class_bases_by_name: dict[str, tuple[str, ...]],
    ) -> Set[str]:
        if isinstance(call.func, PyIRSymLink):
            sid = call.func.symbol_id

            if sid in fn_ret_by_sym:
                return set(fn_ret_by_sym[sid])

            global_map = getattr(ctx, "fn_return_types_by_symbol_id", None)
            if isinstance(global_map, dict):
                v = global_map.get(sid)
                if isinstance(v, frozenset):
                    return set(v)

            return {TYPE_VALUE}

        if isinstance(call.func, PyIRAttribute):
            mname = call.func.attr
            base = call.func.value

            if isinstance(base, PyIRSymLink):
                base_types = env.get(base.symbol_id)
                if not base_types:
                    return {TYPE_VALUE}

                candidates = [
                    t for t in base_types if t not in (TYPE_NONE, TYPE_NIL, TYPE_VALUE)
                ]
                if not candidates:
                    return {TYPE_VALUE}

                out: Set[str] = set()
                found_any = False

                for cls_name in candidates:
                    for owner in TypeFlowPass._iter_class_lineage(
                        cls_name, class_bases_by_name
                    ):
                        ret = method_ret_by_class.get((owner, mname))
                        if ret is None:
                            global_m = getattr(
                                ctx, "method_return_types_by_class", None
                            )
                            if isinstance(global_m, dict):
                                ret = global_m.get((owner, mname))

                        if isinstance(ret, frozenset):
                            out |= set(ret)
                            found_any = True
                            break

                if found_any:
                    return out

                warn_once(
                    f"ret_unknown_{ctx.uid(call)}",
                    f"WARN: Не удалось вывести return-тип для вызова '{mname}()'. "
                    f"Типы receiver: {sorted(base_types)}",
                )
                return {TYPE_VALUE}

        return {TYPE_VALUE}

    @staticmethod
    def _infer_call_return_ann_list(
        ctx: SymLinkContext,
        call: PyIRCall,
        env: Dict[int, Set[str]],
        fn_ret_ann_by_sym: dict[int, str | None],
        method_ret_ann_by_class: dict[tuple[str, str], str | None],
        class_bases_by_name: dict[str, tuple[str, ...]],
    ) -> list[str | None]:
        """
        Best-effort list of possible raw return annotations for call.
        Used only for for-loop element inference.
        """

        if isinstance(call.func, PyIRSymLink):
            sid = call.func.symbol_id
            ann = fn_ret_ann_by_sym.get(sid)

            if ann is None:
                global_map = getattr(ctx, "fn_return_ann_by_symbol_id", None)
                if isinstance(global_map, dict):
                    ann = global_map.get(sid)

            return [ann] if isinstance(ann, (str, type(None))) else []

        if isinstance(call.func, PyIRAttribute):
            mname = call.func.attr
            base = call.func.value

            if isinstance(base, PyIRSymLink):
                base_types = env.get(base.symbol_id) or set()
                candidates = [
                    t for t in base_types if t not in (TYPE_NONE, TYPE_NIL, TYPE_VALUE)
                ]
                out: list[str | None] = []

                global_m = getattr(ctx, "method_return_ann_by_class", None)
                if not isinstance(global_m, dict):
                    global_m = {}

                for cls_name in candidates:
                    for owner in TypeFlowPass._iter_class_lineage(
                        cls_name, class_bases_by_name
                    ):
                        ann = method_ret_ann_by_class.get((owner, mname))
                        if ann is None:
                            ann = global_m.get((owner, mname))

                        if isinstance(ann, (str, type(None))):
                            out.append(ann)
                            break

                return out

        return []

    @staticmethod
    def _iter_class_lineage(
        cls_name: str, class_bases_by_name: dict[str, tuple[str, ...]]
    ) -> Iterable[str]:
        """
        Yields cls_name, then bases in BFS order (best-effort).
        """
        yield cls_name
        seen: set[str] = {cls_name}
        q: list[str] = [cls_name]

        global_bases = getattr(class_bases_by_name, "get", None)

        while q:
            cur = q.pop(0)
            bases = class_bases_by_name.get(cur) or ()
            if not bases:
                pass

            for b in bases:
                if b in seen:
                    continue
                seen.add(b)
                yield b
                q.append(b)

    @staticmethod
    def _narrow_from_test(
        ctx: SymLinkContext,
        test: PyIRNode,
        env: Dict[int, Set[str]],
        warn_once,
        nil_ids: set[int],
        types_alias_ids: set[int],
    ) -> Tuple[Dict[int, Set[str]], Dict[int, Set[str]]]:
        if isinstance(test, PyIRUnaryOP) and test.op == PyUnaryOPType.NOT:
            t_env, f_env = TypeFlowPass._narrow_from_test(
                ctx, test.value, env, warn_once, nil_ids, types_alias_ids
            )
            return f_env, t_env

        guard_arg = TypeFlowPass._match_isvalid(test)
        if guard_arg is not None:
            sym = TypeFlowPass._sym_id(guard_arg)
            if sym is None:
                return env, env

            cur = env.get(sym)
            if cur is None:
                return env, env

            new_t = set(cur)
            new_t.discard(TYPE_NONE)
            new_t.discard(TYPE_NIL)
            if not new_t:
                new_t = {TYPE_VALUE}

            env_true = dict(env)
            env_true[sym] = new_t
            return env_true, env

        if isinstance(test, PyIRBinOP) and test.op in (
            PyBinOPType.IS,
            PyBinOPType.IS_NOT,
            PyBinOPType.EQ,
            PyBinOPType.NE,
        ):
            sym = TypeFlowPass._sym_id(test.left)
            if sym is None:
                return env, env

            rhs_kind = TypeFlowPass._rhs_null_kind(test.right, nil_ids, types_alias_ids)
            if rhs_kind is None:
                return env, env

            cur = env.get(sym)
            if cur is None:
                return env, env

            if rhs_kind == TYPE_NIL and (TYPE_NONE in cur) and (TYPE_NIL not in cur):
                warn_once(
                    f"mismatch_nil_{ctx.uid(test)}",
                    "WARN: Проверка на Nil, но тип содержит None. Сужаем как можем.",
                )

            if rhs_kind == TYPE_NONE and (TYPE_NIL in cur) and (TYPE_NONE not in cur):
                warn_once(
                    f"mismatch_none_{ctx.uid(test)}",
                    "WARN: Проверка на None, но тип содержит Nil. Сужаем как можем.",
                )

            is_not = test.op in (PyBinOPType.IS_NOT, PyBinOPType.NE)

            env_true = dict(env)
            env_false = dict(env)

            if is_not:
                t_true = set(cur)
                t_true.discard(rhs_kind)
                if not t_true:
                    t_true = {TYPE_VALUE}

                env_true[sym] = t_true
                env_false[sym] = {rhs_kind}
                return env_true, env_false

            env_true[sym] = {rhs_kind}
            t_false = set(cur)
            t_false.discard(rhs_kind)
            if not t_false:
                t_false = {TYPE_VALUE}

            env_false[sym] = t_false
            return env_true, env_false

        return env, env

    @staticmethod
    def _match_isvalid(test: PyIRNode) -> PyIRNode | None:
        if not isinstance(test, PyIRCall):
            return None

        if len(test.args_p) != 1:
            return None

        fn = test.func
        if isinstance(fn, PyIRSymLink) and fn.name in TypeFlowPass._GUARD_NAMES:
            return test.args_p[0]

        if isinstance(fn, PyIRVarUse) and fn.name in TypeFlowPass._GUARD_NAMES:
            return test.args_p[0]

        return None

    @staticmethod
    def _sym_id(n: PyIRNode) -> int | None:
        if isinstance(n, PyIRSymLink):
            return n.symbol_id

        return None

    @staticmethod
    def _rhs_null_kind(
        rhs: PyIRNode, nil_ids: set[int], types_alias_ids: set[int]
    ) -> str | None:
        if isinstance(rhs, PyIRConstant) and rhs.value is None:
            return TYPE_NONE

        if TypeFlowPass._rhs_is_nil(rhs, nil_ids, types_alias_ids):
            return TYPE_NIL

        return None

    @staticmethod
    def _rhs_is_nil(
        rhs: PyIRNode, nil_ids: set[int], types_alias_ids: set[int]
    ) -> bool:
        if isinstance(rhs, PyIRConstant):
            try:
                if rhs.value is LuaNil or isinstance(rhs.value, LuaNil):  # type: ignore[arg-type]
                    return True

            except TypeError:
                if rhs.value is LuaNil:
                    return True

        if isinstance(rhs, PyIRSymLink) and rhs.symbol_id in nil_ids:
            return True

        if isinstance(rhs, PyIRCall) and not rhs.args_p and not rhs.args_kw:
            if isinstance(rhs.func, PyIRSymLink) and rhs.func.symbol_id in nil_ids:
                return True

        if isinstance(rhs, PyIRAttribute) and rhs.attr == "nil":
            if (
                isinstance(rhs.value, PyIRSymLink)
                and rhs.value.symbol_id in types_alias_ids
            ):
                return True

        return False

    @staticmethod
    def _merge_env(
        a: Dict[int, Set[str]], b: Dict[int, Set[str]]
    ) -> Dict[int, Set[str]]:
        out: Dict[int, Set[str]] = {}
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            ta = a.get(k)
            tb = b.get(k)
            if ta is None:
                out[k] = set(tb) if tb is not None else {TYPE_VALUE}
                continue

            if tb is None:
                out[k] = set(ta)
                continue

            out[k] = set(ta) | set(tb)

        return out

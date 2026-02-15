from __future__ import annotations

from typing import Final

from ....._cli import CompilerExit
from ....py.ir_dataclass import (
    PyAugAssignType,
    PyBinOPType,
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBinOP,
    PyIRCall,
    PyIRConstant,
    PyIRDict,
    PyIRDictItem,
    PyIREmitExpr,
    PyIREmitKind,
    PyIRFile,
    PyIRFor,
    PyIRList,
    PyIRNode,
    PyIRSubscript,
    PyIRTuple,
    PyIRVarUse,
    PyIRWhile,
)
from ..analysis.symlinks import PyIRSymLink, SymLinkContext
from ..common import (
    CORE_TYPES_MODULES,
    attr_chain_parts,
    collect_local_imported_symbol_ids,
    collect_symbol_ids_in_modules,
    is_expr_on_symbol_ids,
    resolve_symbol_ids_by_attr_chain,
)
from ..rewrite_utils import rewrite_expr_with_store_default, rewrite_stmt_block


class RewriteLuaTableCtorPass:
    """
    Lowering-pass:
      `lua_table(array=[...], map={...})` -> `PyIRDict(...)`.

    Поддерживаемые формы вызова:
      - lua_table()
      - lua_table([1, 2, 3])
      - lua_table([1, 2], {"x": 10})
      - lua_table(array=[...], map={...})
      - lua_table.__init__(...) (на случай уже нормализованного ctor-вызова)
    """

    _TYPES_MODULES: Final[tuple[str, ...]] = CORE_TYPES_MODULES
    _LUA_TABLE_NAME: Final[str] = "lua_table"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        lua_table_symbol_ids = collect_symbol_ids_in_modules(
            ctx,
            symbol_name=cls._LUA_TABLE_NAME,
            modules=cls._TYPES_MODULES,
        )

        if ir.path is not None:
            current_module = ctx.module_name_by_path.get(ir.path)
            if current_module:
                lua_table_symbol_ids |= collect_local_imported_symbol_ids(
                    ir,
                    ctx=ctx,
                    current_module=current_module,
                    imported_name=cls._LUA_TABLE_NAME,
                    allowed_modules=cls._TYPES_MODULES,
                )

        if not lua_table_symbol_ids:
            return ir

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            return [rw_expr(st, False)]

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            if isinstance(node, PyIRCall):
                node.func = rw_expr(node.func, False)
                node.args_p = [rw_expr(a, False) for a in node.args_p]
                node.args_kw = {k: rw_expr(v, False) for k, v in node.args_kw.items()}

                if store:
                    return node

                rewritten = cls._rewrite_ctor_call(
                    ir=ir,
                    ctx=ctx,
                    call=node,
                    lua_table_symbol_ids=lua_table_symbol_ids,
                )
                if rewritten is not None:
                    return rewritten

                return node

            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

    @classmethod
    def _rewrite_ctor_call(
        cls,
        *,
        ir: PyIRFile,
        ctx: SymLinkContext,
        call: PyIRCall,
        lua_table_symbol_ids: set[int],
    ) -> PyIRDict | None:
        if not cls._is_lua_table_ctor_expr(
            expr=call.func,
            ctx=ctx,
            lua_table_symbol_ids=lua_table_symbol_ids,
        ):
            return None

        array_node, map_node = cls._parse_ctor_args(ir, call)
        array_values = cls._normalize_array_arg(ir, call, array_node)
        map_items = cls._normalize_map_arg(ir, call, map_node)

        items: list[PyIRDictItem] = []
        for i, value_node in enumerate(array_values, start=1):
            items.append(
                PyIRDictItem(
                    line=value_node.line,
                    offset=value_node.offset,
                    key=PyIRConstant(
                        line=value_node.line,
                        offset=value_node.offset,
                        value=i,
                    ),
                    value=value_node,
                )
            )

        items.extend(map_items)
        return PyIRDict(
            line=call.line,
            offset=call.offset,
            items=items,
        )

    @classmethod
    def _is_lua_table_ctor_expr(
        cls,
        *,
        expr: PyIRNode,
        ctx: SymLinkContext,
        lua_table_symbol_ids: set[int],
    ) -> bool:
        def is_lua_table_attr_chain(n: PyIRNode) -> bool:
            parts = attr_chain_parts(n)
            if not parts:
                return False
            return (
                tuple(parts[-2:]) == ("glua", "lua_table")
                or tuple(parts[-2:]) == ("core", "lua_table")
                or tuple(parts[-2:]) == ("types", "lua_table")
            )

        def is_lua_table_emit_ref(n: PyIRNode) -> bool:
            if not isinstance(n, PyIREmitExpr):
                return False
            if n.kind == PyIREmitKind.GLOBAL:
                return n.name.endswith(".lua_table") or n.name == "lua_table"
            if n.kind == PyIREmitKind.ATTR:
                return n.name == "lua_table"
            return False

        if isinstance(expr, PyIRAttribute) and expr.attr == "__init__":
            return is_expr_on_symbol_ids(
                expr.value,
                symbol_ids=lua_table_symbol_ids,
                ctx=ctx,
            ) or is_lua_table_attr_chain(expr.value)

        if isinstance(expr, PyIRSymLink):
            return int(expr.symbol_id) in lua_table_symbol_ids

        if is_lua_table_emit_ref(expr):
            return True

        if is_lua_table_attr_chain(expr):
            return True

        resolved_ids = resolve_symbol_ids_by_attr_chain(ctx, expr)
        return any(symbol_id in lua_table_symbol_ids for symbol_id in resolved_ids)

    @classmethod
    def _parse_ctor_args(
        cls,
        ir: PyIRFile,
        call: PyIRCall,
    ) -> tuple[PyIRNode | None, PyIRNode | None]:
        allowed_kw = {"array", "map"}
        for key in call.args_kw:
            if key not in allowed_kw:
                CompilerExit.user_error_node(
                    f"Неизвестный keyword-аргумент '{key}' в lua_table().",
                    ir.path,
                    call,
                )

        if len(call.args_p) > 2:
            CompilerExit.user_error_node(
                "lua_table() ожидает не более 2 позиционных аргументов: (array, map).",
                ir.path,
                call,
            )

        def pick_arg(index: int, key: str) -> PyIRNode | None:
            if index < len(call.args_p):
                if key in call.args_kw:
                    CompilerExit.user_error_node(
                        f"Аргумент '{key}' передан дважды (positional + keyword) в lua_table().",
                        ir.path,
                        call,
                    )
                return call.args_p[index]
            return call.args_kw.get(key)

        array_node = pick_arg(0, "array")
        map_node = pick_arg(1, "map")
        return array_node, map_node

    @classmethod
    def _normalize_array_arg(
        cls,
        ir: PyIRFile,
        call: PyIRCall,
        node: PyIRNode | None,
    ) -> list[PyIRNode]:
        _ = cls
        if node is None:
            return []

        if isinstance(node, PyIRConstant) and node.value is None:
            return []

        if isinstance(node, PyIRList):
            return list(node.elements)

        if isinstance(node, PyIRTuple):
            return list(node.elements)

        CompilerExit.user_error_node(
            "Параметр array в lua_table() должен быть list/tuple или None.",
            ir.path,
            call,
        )
        raise AssertionError("unreachable")

    @classmethod
    def _normalize_map_arg(
        cls,
        ir: PyIRFile,
        call: PyIRCall,
        node: PyIRNode | None,
    ) -> list[PyIRDictItem]:
        _ = cls
        if node is None:
            return []

        if isinstance(node, PyIRConstant) and node.value is None:
            return []

        if isinstance(node, PyIRDict):
            return list(node.items)

        CompilerExit.user_error_node(
            "Параметр map в lua_table() должен быть dict или None.",
            ir.path,
            call,
        )
        raise AssertionError("unreachable")


class RewriteForIteratorStrategyPass:
    """
    Lowering-pass:
      Автоматически подставляет iterator для `for`-циклов.

    Правила:
      - Явный iterator не трогаем:
          pairs(...), ipairs(...), RandomPairs(...), SortedPairs(...), SortedPairsByValue(...)
      - Неявный iterator:
          for value in table_expr:
              -> for __p2g_i_N, value in ipairs(table_expr):
          for key, value in table_expr:
              -> for key, value in pairs(table_expr):
      - Для литерала dict с одной целью:
          for key in {"a": 1} -> pairs(...)
    """

    _TYPES_MODULES: Final[tuple[str, ...]] = CORE_TYPES_MODULES
    _LUA_TABLE_NAME: Final[str] = "lua_table"
    _EXPLICIT_ITERATORS: Final[frozenset[str]] = frozenset(
        {"pairs", "ipairs", "RandomPairs", "SortedPairs", "SortedPairsByValue"}
    )
    _BY_LEN_ITERATOR_METHOD: Final[str] = "by_len"
    _AUTO_VALUE_ITERATOR: Final[str] = "ipairs"
    _AUTO_PAIRS_ITERATOR: Final[str] = "pairs"
    _DICT_ITEMS_METHOD: Final[str] = "items"
    _DICT_KEYS_METHOD: Final[str] = "keys"
    _DICT_VALUES_METHOD: Final[str] = "values"
    _DICT_ITER_METHODS: Final[frozenset[str]] = frozenset(
        {
            _DICT_ITEMS_METHOD,
            _DICT_KEYS_METHOD,
            _DICT_VALUES_METHOD,
        }
    )
    _AUTO_IDX_PREFIX: Final[str] = "__p2g_i_"
    _BY_LEN_TBL_PREFIX: Final[str] = "__p2g_len_tbl_"
    _BY_LEN_IDX_PREFIX: Final[str] = "__p2g_len_i_"

    @classmethod
    def run(cls, ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()
        hidden_idx = 0

        lua_table_symbol_ids = collect_symbol_ids_in_modules(
            ctx,
            symbol_name=cls._LUA_TABLE_NAME,
            modules=cls._TYPES_MODULES,
        )
        if ir.path is not None:
            current_module = ctx.module_name_by_path.get(ir.path)
            if current_module:
                lua_table_symbol_ids |= collect_local_imported_symbol_ids(
                    ir,
                    ctx=ctx,
                    current_module=current_module,
                    imported_name=cls._LUA_TABLE_NAME,
                    allowed_modules=cls._TYPES_MODULES,
                )

        def new_hidden_target(ref: PyIRNode) -> PyIRVarUse:
            nonlocal hidden_idx
            hidden_idx += 1
            return PyIRVarUse(
                line=ref.line,
                offset=ref.offset,
                name=f"{cls._AUTO_IDX_PREFIX}{hidden_idx}",
            )

        def new_tmp_name(prefix: str) -> str:
            nonlocal hidden_idx
            hidden_idx += 1
            return f"{prefix}{hidden_idx}"

        def mk_var(name: str, ref: PyIRNode) -> PyIRVarUse:
            return PyIRVarUse(
                line=ref.line,
                offset=ref.offset,
                name=name,
            )

        def rw_block(body: list[PyIRNode]) -> list[PyIRNode]:
            return rewrite_stmt_block(body, rw_stmt)

        def rw_stmt(st: PyIRNode) -> list[PyIRNode]:
            if isinstance(st, PyIRFor):
                st.target = rw_expr(st.target, True)
                st.iter = rw_expr(st.iter, False)
                st.body = rw_block(st.body)

                by_len_source = cls._extract_by_len_source(
                    ir=ir,
                    ctx=ctx,
                    expr=st.iter,
                    lua_table_symbol_ids=lua_table_symbol_ids,
                )
                if by_len_source is not None:
                    return cls._lower_for_by_len(
                        ir=ir,
                        node=st,
                        source=by_len_source,
                        mk_var=mk_var,
                        new_tmp_name=new_tmp_name,
                    )

                cls._rewrite_for_header(
                    ir=ir,
                    st=st,
                    new_hidden_target=new_hidden_target,
                )
                return [st]

            return [rw_expr(st, False)]

        def rw_expr(node: PyIRNode, store: bool) -> PyIRNode:
            return rewrite_expr_with_store_default(
                node,
                rw_expr=rw_expr,
                rw_block=rw_block,
            )

        ir.body = rw_block(ir.body)
        return ir

    @classmethod
    def _extract_by_len_source(
        cls,
        *,
        ir: PyIRFile,
        ctx: SymLinkContext,
        expr: PyIRNode,
        lua_table_symbol_ids: set[int],
    ) -> PyIRNode | None:
        def is_lua_table_attr_chain(n: PyIRNode) -> bool:
            parts = attr_chain_parts(n)
            if not parts:
                return False
            return (
                tuple(parts[-2:]) == ("glua", "lua_table")
                or tuple(parts[-2:]) == ("core", "lua_table")
                or tuple(parts[-2:]) == ("types", "lua_table")
            )

        def is_lua_table_emit_ref(n: PyIRNode) -> bool:
            if not isinstance(n, PyIREmitExpr):
                return False
            if n.kind == PyIREmitKind.GLOBAL:
                return n.name.endswith(".lua_table") or n.name == "lua_table"
            if n.kind == PyIREmitKind.ATTR:
                return n.name == "lua_table"
            return False

        if not isinstance(expr, PyIRCall):
            return None

        func = expr.func
        if not isinstance(func, PyIRAttribute):
            return None
        if func.attr != cls._BY_LEN_ITERATOR_METHOD:
            return None

        if not (
            is_expr_on_symbol_ids(
                func.value,
                symbol_ids=lua_table_symbol_ids,
                ctx=ctx,
            )
            or is_lua_table_attr_chain(func.value)
            or is_lua_table_emit_ref(func.value)
        ):
            return None

        if expr.args_kw:
            CompilerExit.user_error_node(
                "lua_table.by_len(...) не поддерживает keyword-аргументы.",
                ir.path,
                expr,
            )

        if len(expr.args_p) != 1:
            CompilerExit.user_error_node(
                "lua_table.by_len(...) ожидает ровно 1 позиционный аргумент.",
                ir.path,
                expr,
            )

        return expr.args_p[0]

    @classmethod
    def _lower_for_by_len(
        cls,
        *,
        ir: PyIRFile,
        node: PyIRFor,
        source: PyIRNode,
        mk_var,
        new_tmp_name,
    ) -> list[PyIRNode]:
        pair_target, value_target = cls._split_for_target(node.target)
        is_pair = pair_target is not None

        if is_pair:
            assert pair_target is not None
            if len(pair_target.elements) != 2:
                CompilerExit.user_error_node(
                    "lua_table.by_len(...) поддерживает только цель из 1 или 2 переменных в for.",
                    ir.path,
                    node,
                )
        elif value_target is None:
            CompilerExit.user_error_node(
                "Некорректная цель цикла for для lua_table.by_len(...).",
                ir.path,
                node,
            )

        tbl_name = new_tmp_name(cls._BY_LEN_TBL_PREFIX)
        idx_name = new_tmp_name(cls._BY_LEN_IDX_PREFIX)

        init_tbl = PyIRAssign(
            line=node.line,
            offset=node.offset,
            targets=[mk_var(tbl_name, node)],
            value=source,
        )
        init_idx = PyIRAssign(
            line=node.line,
            offset=node.offset,
            targets=[mk_var(idx_name, node)],
            value=PyIRConstant(line=node.line, offset=node.offset, value=1),
        )

        cond = PyIRBinOP(
            line=node.line,
            offset=node.offset,
            op=PyBinOPType.LE,
            left=mk_var(idx_name, node),
            right=PyIREmitExpr(
                line=node.line,
                offset=node.offset,
                kind=PyIREmitKind.RAW,
                name=f"#{tbl_name}",
                args_p=[],
                args_kw={},
            ),
        )

        cur_value = PyIRSubscript(
            line=node.line,
            offset=node.offset,
            value=mk_var(tbl_name, node),
            index=mk_var(idx_name, node),
        )

        if is_pair:
            assert pair_target is not None
            bind_stmt = PyIRAssign(
                line=node.line,
                offset=node.offset,
                targets=[pair_target],
                value=PyIRTuple(
                    line=node.line,
                    offset=node.offset,
                    elements=[mk_var(idx_name, node), cur_value],
                ),
            )
        else:
            assert value_target is not None
            bind_stmt = PyIRAssign(
                line=node.line,
                offset=node.offset,
                targets=[value_target],
                value=cur_value,
            )

        step_stmt = PyIRAugAssign(
            line=node.line,
            offset=node.offset,
            target=mk_var(idx_name, node),
            op=PyAugAssignType.ADD,
            value=PyIRConstant(line=node.line, offset=node.offset, value=1),
        )

        while_body = [bind_stmt, *node.body, step_stmt]
        while_node = PyIRWhile(
            line=node.line,
            offset=node.offset,
            test=cond,
            body=while_body,
        )

        return [init_tbl, init_idx, while_node]

    @classmethod
    def _rewrite_for_header(
        cls,
        *,
        ir: PyIRFile,
        st: PyIRFor,
        new_hidden_target,
    ) -> None:
        dict_iter = cls._extract_dict_iter_source(st.iter)
        if dict_iter is not None:
            method_name, source = dict_iter
            cls._rewrite_for_dict_iter_method(
                ir=ir,
                st=st,
                method_name=method_name,
                source=source,
                new_hidden_target=new_hidden_target,
            )
            return

        if cls._is_explicit_iterator_expr(st.iter):
            return

        pair_target, value_target = cls._split_for_target(st.target)
        if pair_target is not None:
            st.iter = cls._wrap_iter(st.iter, cls._AUTO_PAIRS_ITERATOR)
            return

        if value_target is None:
            return

        if isinstance(st.iter, PyIRDict):
            st.iter = cls._wrap_iter(st.iter, cls._AUTO_PAIRS_ITERATOR)
            return

        st.iter = cls._wrap_iter(st.iter, cls._AUTO_VALUE_ITERATOR)
        hidden = new_hidden_target(st.target)
        st.target = PyIRTuple(
            line=st.target.line,
            offset=st.target.offset,
            elements=[hidden, value_target],
        )

    @classmethod
    def _extract_dict_iter_source(cls, expr: PyIRNode) -> tuple[str, PyIRNode] | None:
        _ = cls
        if not isinstance(expr, PyIRCall):
            return None
        if expr.args_kw:
            return None
        if expr.args_p:
            return None
        if not isinstance(expr.func, PyIRAttribute):
            return None
        if expr.func.attr not in cls._DICT_ITER_METHODS:
            return None
        return expr.func.attr, expr.func.value

    @classmethod
    def _rewrite_for_dict_iter_method(
        cls,
        *,
        ir: PyIRFile,
        st: PyIRFor,
        method_name: str,
        source: PyIRNode,
        new_hidden_target,
    ) -> None:
        pair_target, value_target = cls._split_for_target(st.target)

        if method_name == cls._DICT_ITEMS_METHOD:
            st.iter = cls._wrap_iter(source, cls._AUTO_PAIRS_ITERATOR)
            if pair_target is not None:
                if len(pair_target.elements) != 2:
                    CompilerExit.user_error_node(
                        "dict.items() в for поддерживает только 2 переменные в цели.",
                        ir.path,
                        st,
                    )
                return

            _ = value_target
            CompilerExit.user_error_node(
                "dict.items() в for требует распаковку: for key, value in dict.items().",
                ir.path,
                st,
            )

        if method_name == cls._DICT_KEYS_METHOD:
            if pair_target is not None:
                CompilerExit.user_error_node(
                    "dict.keys() в for поддерживает только одну переменную в цели.",
                    ir.path,
                    st,
                )
            if value_target is None:
                CompilerExit.user_error_node(
                    "Некорректная цель цикла for для dict.keys().",
                    ir.path,
                    st,
                )
            st.iter = cls._wrap_iter(source, cls._AUTO_PAIRS_ITERATOR)
            st.target = value_target
            return

        if method_name == cls._DICT_VALUES_METHOD:
            if pair_target is not None:
                CompilerExit.user_error_node(
                    "dict.values() в for поддерживает только одну переменную в цели.",
                    ir.path,
                    st,
                )
            if value_target is None:
                CompilerExit.user_error_node(
                    "Некорректная цель цикла for для dict.values().",
                    ir.path,
                    st,
                )
            st.iter = cls._wrap_iter(source, cls._AUTO_PAIRS_ITERATOR)
            hidden = new_hidden_target(st.target)
            st.target = PyIRTuple(
                line=st.target.line,
                offset=st.target.offset,
                elements=[hidden, value_target],
            )
            return

    @classmethod
    def _split_for_target(
        cls,
        target: PyIRNode,
    ) -> tuple[PyIRTuple | PyIRList | None, PyIRNode | None]:
        _ = cls
        if isinstance(target, (PyIRTuple, PyIRList)):
            if not target.elements:
                return None, None
            if len(target.elements) == 1:
                return None, target.elements[0]
            return target, None
        return None, target

    @classmethod
    def _is_explicit_iterator_expr(cls, expr: PyIRNode) -> bool:
        if isinstance(expr, PyIREmitExpr) and expr.kind == PyIREmitKind.CALL:
            return expr.name in cls._EXPLICIT_ITERATORS

        if not isinstance(expr, PyIRCall):
            return False

        name = cls._callable_name(expr.func)
        if name is None:
            return False
        return name in cls._EXPLICIT_ITERATORS

    @classmethod
    def _callable_name(cls, expr: PyIRNode) -> str | None:
        _ = cls
        if isinstance(expr, (PyIRVarUse, PyIRSymLink)):
            return expr.name
        if isinstance(expr, PyIRAttribute):
            return expr.attr
        if isinstance(expr, PyIREmitExpr) and expr.kind == PyIREmitKind.GLOBAL:
            return expr.name
        return None

    @classmethod
    def _wrap_iter(cls, source: PyIRNode, iterator_name: str) -> PyIRCall:
        _ = cls
        return PyIRCall(
            line=source.line,
            offset=source.offset,
            func=PyIREmitExpr(
                line=source.line,
                offset=source.offset,
                kind=PyIREmitKind.GLOBAL,
                name=iterator_name,
                args_p=[],
                args_kw={},
            ),
            args_p=[source],
            args_kw={},
        )

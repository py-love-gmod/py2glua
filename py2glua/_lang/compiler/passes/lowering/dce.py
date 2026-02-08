from __future__ import annotations

from typing import Final, cast

from ....compiler.passes.analysis.symlinks import PyIRSymLink, SymLinkContext
from ....py.ir_dataclass import (
    LuaNil,
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRBinOP,
    PyIRBreak,
    PyIRCall,
    PyIRClassDef,
    PyIRComment,
    PyIRConstant,
    PyIRContinue,
    PyIRDecorator,
    PyIRDict,
    PyIRDictItem,
    PyIREmitExpr,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRIf,
    PyIRImport,
    PyIRList,
    PyIRNode,
    PyIRPass,
    PyIRReturn,
    PyIRSet,
    PyIRSubscript,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarCreate,
    PyIRVarUse,
    PyIRWhile,
    PyIRWith,
    PyIRWithItem,
)
from ...compiler_ir import PyIRDo, PyIRFunctionExpr, PyIRGoto, PyIRLabel


class DcePass:
    """
    Minimal DCE pass (per your rules):

    1) Remove useless expression statements without assignment (pure computations).
    2) If/While with constant test:
         - if const -> replace with reachable branch
         - while const falsy -> remove
    3) Remove unreachable code after:
         - return
         - a call to a NoReturn function
         - goto (until matching label), best-effort

    Does NOT touch `with` semantics (no errors, no validation).
    Should run after NormalizeNilPass + ConstFoldingPass.
    """

    _NORETURN_MARKERS: Final[tuple[str, ...]] = ("NoReturn", "typing.NoReturn")

    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        def is_lua_nil(v: object) -> bool:
            return v is LuaNil or isinstance(v, LuaNil)

        def is_const_value(v: object) -> bool:
            return v is None or is_lua_nil(v) or isinstance(v, (bool, int, float, str))

        def lua_truthy_value(v: object) -> bool:
            return not (v is False or v is None or is_lua_nil(v))

        def const_truthy(node: PyIRNode) -> bool | None:
            if not isinstance(node, PyIRConstant):
                return None

            v = node.value
            if not is_const_value(v):
                return None

            return lua_truthy_value(v)

        def is_useless_expr_stmt(node: PyIRNode) -> bool:
            """
            Only remove expressions that are obviously side-effect-free.

            We keep:
              - PyIRCall (side effects)
              - PyIREmitExpr (side effects)
              - PyIRAttribute / PyIRSubscript (metamethods / __index can be effects in Lua)
            """
            if isinstance(node, (PyIRConstant, PyIRVarUse, PyIRUnaryOP, PyIRBinOP)):
                return True

            if isinstance(node, (PyIRList, PyIRTuple, PyIRSet, PyIRDict, PyIRDictItem)):
                return True

            return False

        def func_is_noreturn_def(fn: PyIRFunctionDef) -> bool:
            if fn.returns is None:
                return False

            r = fn.returns.strip()
            return any(m in r for m in DcePass._NORETURN_MARKERS)

        def build_noreturn_sets_for_file() -> tuple[set[int], set[str]]:
            noreturn_ids: set[int] = set()
            noreturn_names: set[str] = set()

            if ir.path is None:
                return noreturn_ids, noreturn_names

            module_name = ctx.module_name_by_path.get(ir.path)
            if not module_name:
                return noreturn_ids, noreturn_names

            for n in ir.walk():
                if not isinstance(n, PyIRFunctionDef):
                    continue

                if not func_is_noreturn_def(n):
                    continue

                noreturn_names.add(n.name)

                sym = ctx.get_exported_symbol(module_name, n.name)
                if sym is not None:
                    noreturn_ids.add(sym.id.value)

            return noreturn_ids, noreturn_names

        noreturn_symbol_ids, noreturn_names = build_noreturn_sets_for_file()

        def call_is_noreturn(call: PyIRCall) -> bool:
            f = call.func

            if isinstance(f, PyIRSymLink):
                if f.symbol_id in noreturn_symbol_ids:
                    return True

                return f.name in noreturn_names

            if isinstance(f, PyIRVarUse):
                return f.name in noreturn_names

            return False

        def stmt_is_terminator(stmt: PyIRNode) -> tuple[bool, str | None]:
            """
            (terminates_linear_flow, resume_label)
            resume_label is for goto -> matching label.
            """
            if isinstance(stmt, PyIRReturn):
                return True, None

            if isinstance(stmt, PyIRCall) and call_is_noreturn(stmt):
                return True, None

            if isinstance(stmt, PyIRGoto):
                return True, stmt.label

            if isinstance(stmt, (PyIRBreak, PyIRContinue)):
                return True, None

            return False, None

        def dce_expr(expr: PyIRNode, *, store: bool = False) -> PyIRNode:
            if isinstance(
                expr,
                (
                    PyIRConstant,
                    PyIRVarUse,
                    PyIRVarCreate,
                    PyIRImport,
                    PyIRComment,
                    PyIRPass,
                ),
            ):
                return expr

            if isinstance(expr, PyIRSymLink):
                return expr

            if isinstance(expr, PyIRUnaryOP):
                expr.value = dce_expr(expr.value)
                return expr

            if isinstance(expr, PyIRBinOP):
                expr.left = dce_expr(expr.left)
                expr.right = dce_expr(expr.right)
                return expr

            if isinstance(expr, PyIRAttribute):
                expr.value = dce_expr(expr.value, store=False)
                return expr

            if isinstance(expr, PyIRSubscript):
                expr.value = dce_expr(expr.value, store=False)
                expr.index = dce_expr(expr.index, store=False)
                return expr

            if isinstance(expr, PyIRList):
                expr.elements = [dce_expr(e) for e in expr.elements]
                return expr

            if isinstance(expr, PyIRTuple):
                expr.elements = [dce_expr(e) for e in expr.elements]
                return expr

            if isinstance(expr, PyIRSet):
                expr.elements = [dce_expr(e) for e in expr.elements]
                return expr

            if isinstance(expr, PyIRDictItem):
                expr.key = dce_expr(expr.key)
                expr.value = dce_expr(expr.value)
                return expr

            if isinstance(expr, PyIRDict):
                new_items: list[PyIRDictItem] = []
                for item in expr.items:
                    rewritten = dce_expr(item) if isinstance(item, PyIRNode) else item
                    if isinstance(rewritten, PyIRDictItem):
                        new_items.append(rewritten)
                expr.items = new_items
                return expr

            if isinstance(expr, PyIRCall):
                expr.func = dce_expr(expr.func)
                expr.args_p = [dce_expr(a) for a in expr.args_p]
                expr.args_kw = {k: dce_expr(v) for k, v in expr.args_kw.items()}
                return expr

            if isinstance(expr, PyIRDecorator):
                expr.exper = dce_expr(expr.exper)
                expr.args_p = [dce_expr(a) for a in expr.args_p]
                expr.args_kw = {k: dce_expr(v) for k, v in expr.args_kw.items()}
                return expr

            if isinstance(expr, PyIREmitExpr):
                expr.args_p = [dce_expr(a) for a in expr.args_p]
                expr.args_kw = {k: dce_expr(v) for k, v in expr.args_kw.items()}
                return expr

            if isinstance(expr, PyIRFunctionExpr):
                expr.body = dce_stmt_list(expr.body, in_loop=False)
                return expr

            return expr

        def dce_stmt_list(stmts: list[PyIRNode], *, in_loop: bool) -> list[PyIRNode]:
            out: list[PyIRNode] = []

            reachable = True
            resume_labels: set[str] = set()

            for st in stmts:
                if not reachable:
                    if isinstance(st, PyIRLabel) and st.name in resume_labels:
                        reachable = True
                        resume_labels.discard(st.name)
                        out.append(st)

                    continue

                if is_useless_expr_stmt(st):
                    continue

                out.extend(dce_stmt(st, in_loop=in_loop))

                if out:
                    term, resume = stmt_is_terminator(out[-1])
                    if term:
                        reachable = False
                        if resume:
                            resume_labels.add(resume)

            return out

        def dce_stmt(stmt: PyIRNode, *, in_loop: bool) -> list[PyIRNode]:
            if isinstance(stmt, PyIRIf):
                stmt.test = dce_expr(stmt.test)
                t = const_truthy(stmt.test)

                if t is True:
                    branch = dce_stmt_list(stmt.body, in_loop=in_loop)
                    return branch

                if t is False:
                    branch = dce_stmt_list(stmt.orelse, in_loop=in_loop)
                    return branch

                stmt.body = dce_stmt_list(stmt.body, in_loop=in_loop)
                stmt.orelse = dce_stmt_list(stmt.orelse, in_loop=in_loop)

                return [stmt]

            if isinstance(stmt, PyIRWhile):
                stmt.test = dce_expr(stmt.test)
                t = const_truthy(stmt.test)
                if t is False:
                    return []

                stmt.body = dce_stmt_list(stmt.body, in_loop=True)
                return [stmt]

            if isinstance(stmt, PyIRFor):
                stmt.target = dce_expr(stmt.target, store=True)
                stmt.iter = dce_expr(stmt.iter)
                stmt.body = dce_stmt_list(stmt.body, in_loop=True)
                return [stmt]

            if isinstance(stmt, PyIRWith):
                for item in stmt.items:
                    item.context_expr = dce_expr(item.context_expr)
                    if item.optional_vars is not None:
                        item.optional_vars = dce_expr(item.optional_vars, store=True)

                stmt.body = dce_stmt_list(stmt.body, in_loop=in_loop)
                return [stmt]

            if isinstance(stmt, PyIRWithItem):
                stmt.context_expr = dce_expr(stmt.context_expr)
                if stmt.optional_vars is not None:
                    stmt.optional_vars = dce_expr(stmt.optional_vars, store=True)

                return [stmt]

            if isinstance(stmt, PyIRDo):
                stmt.body = dce_stmt_list(stmt.body, in_loop=in_loop)
                if not stmt.body:
                    return []

                return [stmt]

            if isinstance(stmt, PyIRAssign):
                stmt.targets = [dce_expr(t, store=True) for t in stmt.targets]
                stmt.value = dce_expr(stmt.value)
                return [stmt]

            if isinstance(stmt, PyIRAugAssign):
                stmt.target = dce_expr(stmt.target, store=True)
                stmt.value = dce_expr(stmt.value)
                return [stmt]

            if isinstance(stmt, PyIRReturn):
                if stmt.value is not None:
                    stmt.value = dce_expr(stmt.value)
                return [stmt]

            if isinstance(stmt, PyIRCall):
                return [dce_expr(stmt)]

            if isinstance(stmt, (PyIRGoto, PyIRLabel, PyIRBreak, PyIRContinue)):
                return [stmt]

            if isinstance(stmt, PyIRFunctionDef):
                stmt.decorators = [
                    cast(PyIRDecorator, dce_expr(d)) for d in stmt.decorators
                ]
                stmt.body = dce_stmt_list(stmt.body, in_loop=False)
                return [stmt]

            if isinstance(stmt, PyIRClassDef):
                stmt.bases = [dce_expr(b) for b in stmt.bases]
                stmt.decorators = [
                    cast(PyIRDecorator, dce_expr(d)) for d in stmt.decorators
                ]
                stmt.body = dce_stmt_list(stmt.body, in_loop=in_loop)
                return [stmt]

            if isinstance(stmt, PyIREmitExpr):
                return [dce_expr(stmt)]

            if isinstance(
                stmt,
                (
                    PyIRImport,
                    PyIRComment,
                    PyIRPass,
                    PyIRVarCreate,
                    PyIRConstant,
                    PyIRVarUse,
                ),
            ):
                return [stmt]

            return [dce_expr(stmt)]

        ir.body = dce_stmt_list(ir.body, in_loop=False)
        return ir

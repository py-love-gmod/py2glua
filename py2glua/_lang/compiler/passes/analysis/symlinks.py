from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal

from .....config import Py2GluaConfig
from ....py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRAugAssign,
    PyIRCall,
    PyIRClassDef,
    PyIRFile,
    PyIRFor,
    PyIRFunctionDef,
    PyIRImport,
    PyIRNode,
    PyIRVarCreate,
    PyIRVarUse,
    PyIRWithItem,
)
from ...compiler_ir import PyIRFunctionExpr


# IR-нода симлинка
@dataclass
class PyIRSymLink(PyIRNode):
    """
    Результат статического связывания имени.

    - symbol_id: стабильный ID символа (в рамках SymLinkContext)
    - hops: сколько scope-ов вверх до объявления
    - is_store: запись (True) или чтение (False)
    """

    name: str
    symbol_id: int
    hops: int
    is_store: bool = False

    decl_file: Path | None = None
    decl_line: int | None = None

    def walk(self):
        yield self

    def __str__(self) -> str:
        return f"<sym {self.symbol_id}>"


# Модель скоупов/символов
@dataclass(frozen=True)
class ScopeId:
    value: int


@dataclass(frozen=True)
class SymbolId:
    value: int


@dataclass
class Symbol:
    id: SymbolId
    name: str
    scope: ScopeId
    decl_file: Path | None
    decl_line: int | None


@dataclass
class Scope:
    id: ScopeId
    parent: ScopeId | None
    file: Path | None
    kind: Literal["module", "function", "class"]
    defs: dict[str, SymbolId] = field(default_factory=dict)


@dataclass(frozen=True)
class Link:
    target: SymbolId
    hops: int


# Контекст анализа + индекс модулей/экспортов
@dataclass
class SymLinkContext:
    _next_scope: int = 1
    _next_symbol: int = 1
    _next_node: int = 1

    scopes: dict[ScopeId, Scope] = field(default_factory=dict)
    symbols: dict[SymbolId, Symbol] = field(default_factory=dict)

    node_scope: dict[int, ScopeId] = field(default_factory=dict)

    file_scope: dict[int, ScopeId] = field(default_factory=dict)
    fn_body_scope: dict[int, ScopeId] = field(default_factory=dict)
    cls_body_scope: dict[int, ScopeId] = field(default_factory=dict)

    var_links: dict[int, Link] = field(default_factory=dict)
    unresolved_uses: set[int] = field(default_factory=set)

    _node_uid: dict[int, int] = field(default_factory=dict)

    # module index
    _all_paths: set[Path] = field(default_factory=set)
    _module_index_ready: bool = False

    # path -> module.name
    module_name_by_path: dict[Path, str] = field(default_factory=dict)

    # module.name -> { export_name -> SymbolId }
    module_exports: dict[str, dict[str, SymbolId]] = field(default_factory=dict)

    # resolve
    project_deps: dict[str, set[str]] = field(default_factory=dict)
    project_exports: dict[str, set[str]] = field(default_factory=dict)
    project_path_by_module: dict[str, Path] = field(default_factory=dict)

    def uid(self, node: PyIRNode) -> int:
        k = id(node)
        existing = self._node_uid.get(k)
        if existing is not None:
            return existing

        uid = self._next_node
        self._next_node += 1
        self._node_uid[k] = uid
        return uid

    def new_scope(
        self,
        *,
        parent: ScopeId | None,
        file: Path | None,
        kind: Literal["module", "function", "class"],
    ) -> ScopeId:
        sid = ScopeId(self._next_scope)
        self._next_scope += 1
        self.scopes[sid] = Scope(id=sid, parent=parent, file=file, kind=kind)
        return sid

    def ensure_def(
        self,
        *,
        scope: ScopeId,
        name: str,
        decl_file: Path | None,
        decl_line: int | None,
    ) -> SymbolId:
        sc = self.scopes[scope]
        existing = sc.defs.get(name)
        if existing is not None:
            return existing

        sym_id = SymbolId(self._next_symbol)
        self._next_symbol += 1
        self.symbols[sym_id] = Symbol(
            id=sym_id,
            name=name,
            scope=scope,
            decl_file=decl_file,
            decl_line=decl_line,
        )
        sc.defs[name] = sym_id
        return sym_id

    def resolve(self, *, scope: ScopeId, name: str) -> Link | None:
        hops = 0
        cur = scope
        while True:
            sc = self.scopes[cur]
            sym = sc.defs.get(name)
            if sym is not None:
                return Link(target=sym, hops=hops)

            if sc.parent is None:
                return None

            cur = sc.parent
            hops += 1

    # Индексация модулей
    @staticmethod
    def _last_index(parts: tuple[str, ...], name: str) -> int | None:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == name:
                return i

        return None

    def _path_to_module(self, p: Path) -> str | None:
        p = p.resolve()
        parts = p.parts

        # INTERNAL: по последнему py2glua в пути
        idx = self._last_index(parts, "py2glua")
        if idx is not None:
            tail = list(parts[idx:])
            if not tail:
                return None

            last = tail[-1]
            if last.endswith(".py"):
                stem = last[:-3]
                if stem == "__init__":
                    tail = tail[:-1]
                else:
                    tail[-1] = stem

            if not tail:
                return None

            if tail[0] != "py2glua":
                return None

            return ".".join(tail)

        src_root = Py2GluaConfig.source.resolve()
        try:
            rel = p.relative_to(src_root)

        except Exception:
            return None

        tail = list(rel.parts)
        if not tail:
            return None

        last = tail[-1]
        if last.endswith(".py"):
            stem = last[:-3]
            if stem == "__init__":
                tail = tail[:-1]
            else:
                tail[-1] = stem

        if not tail:
            return None

        return ".".join(tail)

    def ensure_module_index(self) -> None:
        if self._module_index_ready:
            return

        for p in sorted(self._all_paths):
            mod = self._path_to_module(p)
            if mod is None:
                continue

            self.module_name_by_path[p] = mod
            self.module_exports.setdefault(mod, {})

        self._module_index_ready = True

    def get_exported_symbol(self, module_path: str, symbol: str) -> Symbol | None:
        """
        Получить экспортируемый символ из модуля.
        """
        self.ensure_module_index()

        exports = self.module_exports.get(module_path)
        if not exports:
            return None

        sym_id = exports.get(symbol)
        if sym_id is None:
            return None

        return self.symbols.get(sym_id)


# Pass 1: построение скоупов
class BuildScopesPass:
    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        if ir.path is not None:
            ctx._all_paths.add(ir.path)

        for n in ir.walk():
            ctx.uid(n)

        file_scope = ctx.new_scope(parent=None, file=ir.path, kind="module")
        file_uid = ctx.uid(ir)
        ctx.node_scope[file_uid] = file_scope
        ctx.file_scope[file_uid] = file_scope

        def iter_nodes(val):
            if isinstance(val, PyIRNode):
                yield val
                return

            if isinstance(val, (list, tuple)):
                for x in val:
                    yield from iter_nodes(x)

                return

            if isinstance(val, dict):
                for k, v in val.items():
                    yield from iter_nodes(k)
                    yield from iter_nodes(v)

                return

        def iter_children_direct(node: PyIRNode):
            if not is_dataclass(node):
                return

            for f in fields(node):
                v = getattr(node, f.name)
                yield from iter_nodes(v)

        def walk(node: PyIRNode, scope: ScopeId) -> None:
            node_uid = ctx.uid(node)
            ctx.node_scope[node_uid] = scope

            if isinstance(node, PyIRFunctionDef):
                for _ann, default in node.signature.values():
                    if default is not None:
                        walk(default, scope)

                body_scope = ctx.new_scope(parent=scope, file=ir.path, kind="function")
                ctx.fn_body_scope[node_uid] = body_scope

                for d in node.decorators:
                    walk(d, scope)

                for b in node.body:
                    walk(b, body_scope)

                return

            if isinstance(node, PyIRFunctionExpr):
                for _ann, default in node.signature.values():
                    if default is not None:
                        walk(default, scope)

                body_scope = ctx.new_scope(parent=scope, file=ir.path, kind="function")
                ctx.fn_body_scope[node_uid] = body_scope

                for b in node.body:
                    walk(b, body_scope)

                return

            if isinstance(node, PyIRClassDef):
                body_scope = ctx.new_scope(parent=scope, file=ir.path, kind="class")
                ctx.cls_body_scope[node_uid] = body_scope

                for b in node.bases:
                    walk(b, scope)

                for d in node.decorators:
                    walk(d, scope)

                for n2 in node.body:
                    walk(n2, body_scope)

                return

            for ch in iter_children_direct(node):
                walk(ch, scope)

        for n in ir.body:
            walk(n, file_scope)


# Pass 2: сбор определений + exports
class CollectDefsPass:
    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        ctx.ensure_module_index()

        file_uid = ctx.uid(ir)
        module_scope = ctx.file_scope[file_uid]
        module_name = (
            ctx.module_name_by_path.get(ir.path) if ir.path is not None else None
        )

        def scope_of(node: PyIRNode) -> ScopeId:
            return ctx.node_scope[ctx.uid(node)]

        def define(scope: ScopeId, name: str, node: PyIRNode) -> SymbolId:
            sym_id = ctx.ensure_def(
                scope=scope,
                name=name,
                decl_file=ir.path,
                decl_line=node.line,
            )
            if module_name is not None and scope == module_scope:
                ctx.module_exports.setdefault(module_name, {})[name] = sym_id

            return sym_id

        def define_if_simple_name(scope: ScopeId, n: PyIRNode) -> None:
            if isinstance(n, PyIRVarUse):
                define(scope, n.name, n)

        for node in ir.walk():
            sc = scope_of(node)

            if isinstance(node, PyIRVarCreate):
                define(sc, node.name, node)
                continue

            if isinstance(node, PyIRFunctionDef):
                define(sc, node.name, node)
                fn_uid = ctx.uid(node)
                body_scope = ctx.fn_body_scope.get(fn_uid, sc)
                for arg in node.signature.keys():
                    define(body_scope, arg, node)

                continue

            if isinstance(node, PyIRFunctionExpr):
                fn_uid = ctx.uid(node)
                body_scope = ctx.fn_body_scope.get(fn_uid, sc)
                for arg in node.signature.keys():
                    define(body_scope, arg, node)

                continue

            if isinstance(node, PyIRClassDef):
                define(sc, node.name, node)
                continue

            if isinstance(node, PyIRAssign):
                for t in node.targets:
                    define_if_simple_name(sc, t)

                continue

            if isinstance(node, PyIRAugAssign):
                define_if_simple_name(sc, node.target)
                continue

            if isinstance(node, PyIRFor):
                define_if_simple_name(sc, node.target)
                continue

            if isinstance(node, PyIRWithItem):
                if node.optional_vars is not None:
                    define_if_simple_name(sc, node.optional_vars)

                continue

            if isinstance(node, PyIRImport):
                if node.names:
                    for n in node.names:
                        name = n[1] if isinstance(n, tuple) else n
                        define(sc, name, node)

                elif node.modules:
                    define(sc, node.modules[0], node)

                continue


# Pass 3: resolve VarUse
class ResolveUsesPass:
    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> None:
        for node in ir.walk():
            if not isinstance(node, PyIRVarUse):
                continue

            uid = ctx.uid(node)
            sc = ctx.node_scope[uid]

            link = ctx.resolve(scope=sc, name=node.name)
            if link is None:
                ctx.unresolved_uses.add(uid)
                continue

            ctx.var_links[uid] = link


# Pass 4: rewrite + collapse attrs to exports
class RewriteToSymlinksPass:
    @staticmethod
    def run(ir: PyIRFile, ctx: SymLinkContext) -> PyIRFile:
        ctx.ensure_module_index()

        def sym_from_symbol_id(
            sym_id: SymbolId,
            *,
            line: int | None,
            offset: int | None,
            store: bool,
        ) -> PyIRSymLink:
            sym = ctx.symbols[sym_id]
            return PyIRSymLink(
                line=line,
                offset=offset,
                name=sym.name,
                symbol_id=sym.id.value,
                hops=0,
                is_store=store,
                decl_file=sym.decl_file,
                decl_line=sym.decl_line,
            )

        def base_name(n: PyIRNode) -> str | None:
            if isinstance(n, PyIRVarUse):
                return n.name

            if isinstance(n, PyIRSymLink):
                return n.name

            return None

        def flatten_attr_chain(n: PyIRAttribute) -> tuple[PyIRNode, list[str]]:
            attrs_rev: list[str] = []
            cur: PyIRNode = n
            while isinstance(cur, PyIRAttribute):
                attrs_rev.append(cur.attr)
                cur = cur.value
            attrs_rev.reverse()
            return cur, attrs_rev

        def try_collapse_attr(n: PyIRAttribute) -> PyIRNode:
            base, attrs = flatten_attr_chain(n)
            bn = base_name(base)
            if bn is None:
                return n

            parts = [bn, *attrs]
            if len(parts) < 2:
                return n

            for j in range(len(parts) - 1, 0, -1):
                module = ".".join(parts[:j])
                export = parts[j]
                rest = parts[j + 1 :]

                exports = ctx.module_exports.get(module)
                if not exports:
                    continue

                target = exports.get(export)
                if target is None:
                    continue

                next_module = f"{module}.{export}"
                if rest and next_module in ctx.module_exports:
                    continue

                cur2: PyIRNode = sym_from_symbol_id(
                    target,
                    line=n.line,
                    offset=n.offset,
                    store=False,
                )

                for a in rest:
                    cur2 = PyIRAttribute(
                        line=n.line,
                        offset=n.offset,
                        value=cur2,
                        attr=a,
                    )

                return cur2

            return n

        def rw(node: PyIRNode, *, store: bool) -> PyIRNode:
            if isinstance(node, PyIRVarUse):
                uid = ctx.uid(node)
                link = ctx.var_links.get(uid)
                if link is None:
                    return node

                sym = ctx.symbols[link.target]
                return PyIRSymLink(
                    line=node.line,
                    offset=node.offset,
                    name=node.name,
                    symbol_id=sym.id.value,
                    hops=link.hops,
                    is_store=store,
                    decl_file=sym.decl_file,
                    decl_line=sym.decl_line,
                )

            if isinstance(node, PyIRAttribute):
                node.value = rw(node.value, store=False)
                return try_collapse_attr(node)

            if isinstance(node, PyIRCall):
                node.func = rw(node.func, store=False)
                node.args_p = [rw(a, store=False) for a in node.args_p]
                node.args_kw = {k: rw(v, store=False) for k, v in node.args_kw.items()}
                return node

            if isinstance(node, PyIRAssign):
                node.targets = [rw(t, store=True) for t in node.targets]
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRAugAssign):
                node.target = rw(node.target, store=True)
                node.value = rw(node.value, store=False)
                return node

            if isinstance(node, PyIRFor):
                node.target = rw(node.target, store=True)
                node.iter = rw(node.iter, store=False)
                node.body = [rw(n, store=False) for n in node.body]
                return node

            if isinstance(node, PyIRWithItem):
                node.context_expr = rw(node.context_expr, store=False)
                if node.optional_vars is not None:
                    node.optional_vars = rw(node.optional_vars, store=True)
                return node

            if isinstance(node, PyIRFunctionDef):
                node.decorators = [rw(d, store=False) for d in node.decorators]  # type: ignore

                new_sig: dict[str, tuple[str | None, PyIRNode | None]] = {}
                for k, (ann, default) in node.signature.items():
                    if default is not None:
                        default = rw(default, store=False)

                    new_sig[k] = (ann, default)

                node.signature = new_sig

                node.body = [rw(n, store=False) for n in node.body]
                return node

            if isinstance(node, PyIRFunctionExpr):
                new_sig: dict[str, tuple[str | None, PyIRNode | None]] = {}
                for k, (ann, default) in node.signature.items():
                    if default is not None:
                        default = rw(default, store=False)

                    new_sig[k] = (ann, default)

                node.signature = new_sig

                node.body = [rw(n, store=False) for n in node.body]
                return node

            if isinstance(node, PyIRClassDef):
                node.decorators = [rw(d, store=False) for d in node.decorators]  # type: ignore
                node.bases = [rw(b, store=False) for b in node.bases]
                node.body = [rw(n, store=False) for n in node.body]
                return node

            if is_dataclass(node):
                for f in fields(node):
                    v = getattr(node, f.name)

                    if isinstance(v, PyIRNode):
                        setattr(node, f.name, rw(v, store=False))

                    elif isinstance(v, list):
                        setattr(
                            node,
                            f.name,
                            [
                                rw(x, store=False) if isinstance(x, PyIRNode) else x
                                for x in v
                            ],
                        )

                    elif isinstance(v, tuple):
                        out_t = []
                        changed = False
                        for x in v:
                            if isinstance(x, PyIRNode):
                                nx = rw(x, store=False)
                                changed = changed or (nx is not x)
                                out_t.append(nx)

                            else:
                                out_t.append(x)

                        if changed:
                            setattr(node, f.name, tuple(out_t))

                    elif isinstance(v, dict):
                        setattr(
                            node,
                            f.name,
                            {
                                k: (
                                    rw(x, store=False) if isinstance(x, PyIRNode) else x
                                )
                                for k, x in v.items()
                            },
                        )

                return node

            return node

        ir.body = [rw(n, store=False) for n in ir.body]
        return ir

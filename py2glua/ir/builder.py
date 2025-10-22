import ast
from typing import Iterable

from .ir_base import (
    Attribute,
    BinOp,
    BinOpType,
    BoolOp,
    BoolOpType,
    Break,
    Call,
    ClassDef,
    Compare,
    Constant,
    Continue,
    DictLiteral,
    ExceptHandler,
    File,
    For,
    FunctionDef,
    If,
    Import,
    ImportType,
    IRNode,
    ListLiteral,
    Pass,
    Return,
    Subscript,
    Try,
    TupleLiteral,
    UnaryOp,
    UnaryOpType,
    VarLoad,
    VarStore,
    While,
    With,
)


class IRBuilder:
    # region Entry
    @classmethod
    def build_ir(cls, module: ast.Module) -> File:
        file = File(lineno=None, col_offset=None, parent=None, file=None)
        for stmt in module.body:
            ir_nodes = cls.build_node(stmt)
            if not ir_nodes:
                continue

            if isinstance(ir_nodes, list):
                cls._append_children(file, ir_nodes)

            else:
                cls._append_child(file, ir_nodes)

        for node in file.walk():
            if node is file:
                continue

            else:
                node.file = file

        return file

    @classmethod
    def build_node(cls, node: ast.AST | None):
        if node is None:
            return None

        handler = getattr(cls, f"build_{type(node).__name__}", None)
        if handler is None:
            raise NotImplementedError(f"Unsupported AST node: {type(node).__name__}")

        return handler(node)

    @staticmethod
    def _append_child(
        parent: File | FunctionDef | If | While | For | With | Try,
        child: IRNode,
    ) -> None:
        child.parent = parent
        parent.body.append(child)

    @staticmethod
    def _append_children(
        parent: File | FunctionDef | If | While | For | With | Try,
        children: Iterable[IRNode],
    ) -> None:
        for ch in children:
            ch.parent = parent
            parent.body.append(ch)

    # endregion

    # region Imports
    @staticmethod
    def build_Import(node: ast.Import) -> Import:
        names = [alias.name for alias in node.names]
        aliases = [alias.asname for alias in node.names]
        return Import(
            module=None,
            names=names,
            aliases=aliases,
            import_type=ImportType.UNKNOWN,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

    @staticmethod
    def build_ImportFrom(node: ast.ImportFrom) -> Import:
        names = [alias.name for alias in node.names]
        aliases = [alias.asname for alias in node.names]
        return Import(
            module=node.module,
            names=names,
            aliases=aliases,
            import_type=ImportType.UNKNOWN,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

    # endregion

    # region Constants / Vars / Assignments
    @staticmethod
    def build_Name(node: ast.Name) -> VarLoad:
        return VarLoad(
            name=node.id,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

    @staticmethod
    def build_Constant(node: ast.Constant) -> Constant:
        return Constant(
            value=node.value,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

    @classmethod
    def build_Assign(cls, node: ast.Assign) -> list[VarStore]:
        value_ir = cls.build_node(node.value)
        if not isinstance(value_ir, IRNode):
            raise NotImplementedError(
                f"Assign value of type {type(value_ir).__name__} is not supported yet"
            )

        stores: list[VarStore] = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                store = VarStore(
                    name=target.id,
                    value=value_ir,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    parent=None,
                    file=None,
                )
                value_ir.parent = store
                stores.append(store)

            else:
                raise NotImplementedError(
                    f"Unsupported assignment target: {type(target).__name__}"
                )

        return stores

    @classmethod
    def build_AnnAssign(cls, node: ast.AnnAssign) -> VarStore:
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(
                f"Unsupported annotated target: {type(node.target).__name__}"
            )

        if node.value is None:
            raise NotImplementedError(
                "Annotated assignment without value is not supported yet"
            )

        value_ir = cls.build_node(node.value)
        assert isinstance(value_ir, IRNode)

        store = VarStore(
            name=node.target.id,
            value=value_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        value_ir.parent = store
        store.annotation = ast.unparse(node.annotation)
        return store

    @classmethod
    def build_AugAssign(cls, node: ast.AugAssign) -> list[VarStore]:
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(
                f"Unsupported augmented target: {type(node.target).__name__}"
            )

        left = VarLoad(
            name=node.target.id,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        right = cls.build_node(node.value)
        assert isinstance(right, IRNode)

        binop = BinOp(
            op=cls._op_to_str(node.op),
            left=left,
            right=right,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        left.parent = binop
        right.parent = binop

        store = VarStore(
            name=node.target.id,
            value=binop,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        binop.parent = store
        return [store]

    # endregion

    # region Expressions
    @classmethod
    def build_BinOp(cls, node: ast.BinOp) -> BinOp:
        left = cls.build_node(node.left)
        right = cls.build_node(node.right)
        assert isinstance(left, IRNode)
        assert isinstance(right, IRNode)
        binop = BinOp(
            op=cls._op_to_str(node.op),
            left=left,
            right=right,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        left.parent = binop
        right.parent = binop
        return binop

    @classmethod
    def build_BoolOp(cls, node: ast.BoolOp) -> BoolOp:
        op_map = {ast.And: BoolOpType.AND, ast.Or: BoolOpType.OR}
        op_type = op_map[type(node.op)]
        left = cls.build_node(node.values[0])
        right = cls.build_node(node.values[1])
        return BoolOp(
            op=op_type,
            left=left,
            right=right,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

    @classmethod
    def build_UnaryOp(cls, node: ast.UnaryOp) -> UnaryOp:
        op_map = {
            ast.UAdd: UnaryOpType.POS,
            ast.USub: UnaryOpType.NEG,
            ast.Not: UnaryOpType.NOT,
            ast.Invert: UnaryOpType.BIT_NOT,
        }
        op_type = op_map.get(type(node.op))
        if op_type is None:
            raise NotImplementedError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )

        operand_ir = cls.build_node(node.operand)
        assert isinstance(operand_ir, IRNode)

        unop_ir = UnaryOp(
            op=op_type,
            operand=operand_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        operand_ir.parent = unop_ir
        return unop_ir

    @classmethod
    def build_Compare(cls, node: ast.Compare) -> Compare:
        op_map = {
            ast.Eq: BoolOpType.EQ,
            ast.NotEq: BoolOpType.NE,
            ast.Lt: BoolOpType.LT,
            ast.LtE: BoolOpType.LE,
            ast.Gt: BoolOpType.GT,
            ast.GtE: BoolOpType.GE,
            ast.Is: BoolOpType.IS,
            ast.IsNot: BoolOpType.IS_NOT,
            ast.In: BoolOpType.IN,
            ast.NotIn: BoolOpType.NOT_IN,
        }
        op_type = op_map.get(type(node.ops[0]))
        if op_type is None:
            raise NotImplementedError(
                f"Unsupported compare operator: {type(node.ops[0]).__name__}"
            )

        left = cls.build_node(node.left)
        right = cls.build_node(node.comparators[0])
        assert isinstance(left, IRNode)
        assert isinstance(right, IRNode)

        cmp_ir = Compare(
            op=op_type,
            left=left,
            right=right,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        left.parent = cmp_ir
        right.parent = cmp_ir
        return cmp_ir

    @classmethod
    def build_Call(cls, node: ast.Call) -> Call:
        func_ir = cls.build_node(node.func)
        assert isinstance(func_ir, IRNode)

        args_ir: list[IRNode] = []
        for arg in node.args:
            arg_ir = cls.build_node(arg)
            assert isinstance(arg_ir, IRNode)
            args_ir.append(arg_ir)

        call_ir = Call(
            func=func_ir,
            args=args_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

        func_ir.parent = call_ir
        for arg in args_ir:
            arg.parent = call_ir

        return call_ir

    @classmethod
    def build_Attribute(cls, node: ast.Attribute) -> Attribute:
        value_ir = cls.build_node(node.value)
        assert isinstance(value_ir, IRNode)
        attr_ir = Attribute(
            value=value_ir,
            attr=node.attr,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        value_ir.parent = attr_ir
        return attr_ir

    @classmethod
    def build_Subscript(cls, node: ast.Subscript) -> Subscript:
        value_ir = cls.build_node(node.value)
        index_ir = cls.build_node(node.slice)
        assert isinstance(value_ir, IRNode)
        assert isinstance(index_ir, IRNode)
        sub_ir = Subscript(
            value=value_ir,
            index=index_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        value_ir.parent = sub_ir
        index_ir.parent = sub_ir
        return sub_ir

    # endregion

    # region Container
    @classmethod
    def build_List(cls, node: ast.List) -> ListLiteral:
        elements = []
        for elt in node.elts:
            ir = cls.build_node(elt)
            assert isinstance(ir, IRNode)
            elements.append(ir)

        list_ir = ListLiteral(
            elements=elements,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        for el in elements:
            el.parent = list_ir

        return list_ir

    @classmethod
    def build_Tuple(cls, node: ast.Tuple) -> TupleLiteral:
        elements = []
        for elt in node.elts:
            ir = cls.build_node(elt)
            assert isinstance(ir, IRNode)
            elements.append(ir)

        tup_ir = TupleLiteral(
            elements=elements,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        for el in elements:
            el.parent = tup_ir

        return tup_ir

    @classmethod
    def build_Dict(cls, node: ast.Dict) -> DictLiteral:
        keys, values = [], []
        for k, v in zip(node.keys, node.values):
            key_ir = cls.build_node(k) if k is not None else None
            val_ir = cls.build_node(v)
            if key_ir:
                assert isinstance(key_ir, IRNode)
                key_ir.parent = None
                keys.append(key_ir)

            assert isinstance(val_ir, IRNode)
            values.append(val_ir)

        dict_ir = DictLiteral(
            keys=keys,
            values=values,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        for el in keys + values:
            el.parent = dict_ir

        return dict_ir

    # endregion

    # region Functions
    @classmethod
    def build_FunctionDef(cls, node: ast.FunctionDef) -> FunctionDef:
        args = [a.arg for a in node.args.args]
        decorators = [ast.unparse(d) for d in node.decorator_list]

        fn = FunctionDef(
            name=node.name,
            args=args,
            decorators=decorators,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
            body=[],
        )

        for stmt in node.body:
            ir = cls.build_node(stmt)
            if not ir:
                continue
            if isinstance(ir, list):
                for ch in ir:
                    ch.parent = fn
                fn.body.extend(ir)
            else:
                ir.parent = fn
                fn.body.append(ir)
        return fn

    @classmethod
    def build_Return(cls, node: ast.Return) -> Return:
        value_ir = cls.build_node(node.value) if node.value else None
        ret = Return(
            value=value_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        if value_ir:
            value_ir.parent = ret

        return ret

    # endregion

    # region Classes
    @classmethod
    def build_ClassDef(cls, node: ast.ClassDef):
        name = node.name
        bases_ir: list[IRNode] = []
        for base in node.bases:
            base_ir = cls.build_node(base)
            assert isinstance(base_ir, IRNode)
            bases_ir.append(base_ir)

        decorators = [ast.unparse(d) for d in node.decorator_list]
        class_ir = ClassDef(
            name=name,
            bases=bases_ir,
            decorators=decorators,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
            body=[],
        )

        for base_ir in bases_ir:
            base_ir.parent = class_ir

        for stmt in node.body:
            stmt_ir = cls.build_node(stmt)
            if not stmt_ir:
                continue

            if isinstance(stmt_ir, list):
                for ch in stmt_ir:
                    ch.parent = class_ir

                class_ir.body.extend(stmt_ir)

            else:
                stmt_ir.parent = class_ir
                class_ir.body.append(stmt_ir)

        return class_ir

    # endregion

    # region Control Flow
    @classmethod
    def build_If(cls, node: ast.If) -> If:
        test_ir = cls.build_node(node.test)
        assert isinstance(test_ir, IRNode)

        if_ir = If(
            test=test_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        test_ir.parent = if_ir

        for stmt in node.body:
            body_ir = cls.build_node(stmt)
            if not body_ir:
                continue

            if isinstance(body_ir, list):
                for ch in body_ir:
                    ch.parent = if_ir

                if_ir.body.extend(body_ir)

            else:
                body_ir.parent = if_ir
                if_ir.body.append(body_ir)

        for stmt in node.orelse:
            orelse_ir = cls.build_node(stmt)
            if not orelse_ir:
                continue

            if isinstance(orelse_ir, list):
                for ch in orelse_ir:
                    ch.parent = if_ir

                if_ir.orelse.extend(orelse_ir)

            else:
                orelse_ir.parent = if_ir
                if_ir.orelse.append(orelse_ir)

        return if_ir

    @classmethod
    def build_While(cls, node: ast.While) -> While:
        test_ir = cls.build_node(node.test)
        assert isinstance(test_ir, IRNode)

        while_ir = While(
            test=test_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        test_ir.parent = while_ir

        for stmt in node.body:
            body_ir = cls.build_node(stmt)
            if not body_ir:
                continue

            cls._append_child(while_ir, body_ir) if not isinstance(
                body_ir, list
            ) else cls._append_children(while_ir, body_ir)

        for stmt in node.orelse:
            orelse_ir = cls.build_node(stmt)
            if not orelse_ir:
                continue

            cls._append_child(while_ir, orelse_ir) if not isinstance(
                orelse_ir, list
            ) else cls._append_children(while_ir, orelse_ir)

        return while_ir

    @classmethod
    def build_For(cls, node: ast.For) -> For:
        target_ir = cls.build_node(node.target)
        iter_ir = cls.build_node(node.iter)
        assert isinstance(target_ir, IRNode)
        assert isinstance(iter_ir, IRNode)

        for_ir = For(
            target=target_ir,
            iter=iter_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )
        target_ir.parent = for_ir
        iter_ir.parent = for_ir

        for stmt in node.body:
            body_ir = cls.build_node(stmt)
            if not body_ir:
                continue

            cls._append_child(for_ir, body_ir) if not isinstance(
                body_ir, list
            ) else cls._append_children(for_ir, body_ir)

        for stmt in node.orelse:
            orelse_ir = cls.build_node(stmt)
            if not orelse_ir:
                continue

            cls._append_child(for_ir, orelse_ir) if not isinstance(
                orelse_ir, list
            ) else cls._append_children(for_ir, orelse_ir)

        return for_ir

    @classmethod
    def build_With(cls, node: ast.With) -> With:
        item = node.items[0]

        context_ir = cls.build_node(item.context_expr)
        assert isinstance(context_ir, IRNode)

        target_ir = None
        if item.optional_vars:
            target_ir = cls.build_node(item.optional_vars)
            assert isinstance(target_ir, IRNode)

        with_ir = With(
            context=context_ir,
            target=target_ir,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
            body=[],
        )
        context_ir.parent = with_ir
        if target_ir:
            target_ir.parent = with_ir

        for stmt in node.body:
            ir = cls.build_node(stmt)
            if not ir:
                continue

            if isinstance(ir, list):
                cls._append_children(with_ir, ir)

            else:
                cls._append_child(with_ir, ir)

        return with_ir

    @classmethod
    def build_Try(cls, node: ast.Try) -> Try:
        try_ir = Try(
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
            body=[],
            handlers=[],
            orelse=[],
            finalbody=[],
        )

        for stmt in node.body:
            ir = cls.build_node(stmt)
            if not ir:
                continue

            if isinstance(ir, list):
                cls._append_children(try_ir, ir)

            else:
                cls._append_child(try_ir, ir)

        for h in node.handlers:
            type_ir = cls.build_node(h.type) if h.type is not None else None
            if type_ir is not None:
                assert isinstance(type_ir, IRNode)

            name_str = (
                h.name if isinstance(h.name, str) or h.name is None else str(h.name)
            )

            handler_ir = ExceptHandler(
                type=type_ir,
                name=name_str,
                lineno=h.lineno,
                col_offset=h.col_offset,
                parent=try_ir,
                file=None,
                body=[],
            )
            if type_ir:
                type_ir.parent = handler_ir

            for stmt in h.body:
                ir = cls.build_node(stmt)
                if not ir:
                    continue

                if isinstance(ir, list):
                    for ch in ir:
                        ch.parent = handler_ir
                        handler_ir.body.append(ch)

                else:
                    ir.parent = handler_ir
                    handler_ir.body.append(ir)

            try_ir.handlers.append(handler_ir)

        for stmt in node.orelse:
            ir = cls.build_node(stmt)
            if not ir:
                continue

            if isinstance(ir, list):
                for ch in ir:
                    ch.parent = try_ir
                    try_ir.orelse.append(ch)

            else:
                ir.parent = try_ir
                try_ir.orelse.append(ir)

        for stmt in node.finalbody:
            ir = cls.build_node(stmt)
            if not ir:
                continue

            if isinstance(ir, list):
                for ch in ir:
                    ch.parent = try_ir
                    try_ir.finalbody.append(ch)

            else:
                ir.parent = try_ir
                try_ir.finalbody.append(ir)

        return try_ir

    @staticmethod
    def build_Break(node: ast.Break) -> Break:
        return Break(
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

    @staticmethod
    def build_Continue(node: ast.Continue) -> Continue:
        return Continue(
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

    @staticmethod
    def build_Pass(node: ast.Pass) -> Pass:
        return Pass(
            lineno=node.lineno,
            col_offset=node.col_offset,
            parent=None,
            file=None,
        )

    # endregion

    # region Misc
    @classmethod
    def build_Expr(cls, node: ast.Expr) -> IRNode:
        value_ir = cls.build_node(node.value)
        assert isinstance(value_ir, IRNode)
        return value_ir

    # endregion

    # region Helpers
    @staticmethod
    def _op_to_str(op: ast.AST) -> BinOpType:
        mapping = {
            ast.Add: BinOpType.ADD,
            ast.Sub: BinOpType.SUB,
            ast.Mult: BinOpType.MUL,
            ast.Div: BinOpType.DIV,
            ast.Mod: BinOpType.MOD,
            ast.BitOr: BinOpType.BIT_OR,
            ast.BitAnd: BinOpType.BIT_AND,
            ast.BitXor: BinOpType.BIT_XOR,
            ast.LShift: BinOpType.BIT_LSHIFT,
            ast.RShift: BinOpType.BIT_RSHIFT,
            ast.FloorDiv: BinOpType.FLOOR_DIV,
            ast.Pow: BinOpType.POW,
        }
        for k, v in mapping.items():
            if isinstance(op, k):
                return v

        raise NotImplementedError(f"Unsupported binary operator: {type(op).__name__}")

    # endregion

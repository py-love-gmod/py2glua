from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from ...._cli.logging_setup import exit_with_code
from ...py.ir_dataclass import (
    PyIRAssign,
    PyIRAttribute,
    PyIRBinOP,
    PyIRCall,
    PyIRClassDef,
    PyIRConstant,
    PyIRDecorator,
    PyIRDict,
    PyIRDictItem,
    PyIRFile,
    PyIRFString,
    PyIRFunctionDef,
    PyIRList,
    PyIRNode,
    PyIRSet,
    PyIRSubscript,
    PyIRTuple,
    PyIRUnaryOP,
    PyIRVarCreate,
    PyIRVarUse,
)

_MethodKind = Literal["instance", "class", "static"]


@dataclass(frozen=True, slots=True)
class _LoweredMethod:
    kind: _MethodKind
    fn: PyIRFunctionDef


@dataclass(frozen=True, slots=True)
class _LoweredClassParts:
    fields: list[PyIRAssign]
    methods: list[_LoweredMethod]
    field_names: set[str]


class LowerClassTablePass:
    """
    Project-pass: понижает Python-классы в Lua-таблицы.

    Поддержка методов:
      - instance method
      - @classmethod
      - @staticmethod

    Важно:
      - Поля класса исполняются в "пространстве имён класса".
        Поэтому любые ссылки на другие поля в RHS должны стать ClassName.<field>.
      - Допускаются только простые поля:
          name = expr
        и ограниченная распаковка:
          a, b = (1, 2)
          a, b = [1, 2]
        (только literal tuple/list, длины должны совпадать)
    """

    @classmethod
    def run(cls, files: list[PyIRFile]) -> list[PyIRFile]:
        for f in files:
            if f.body:
                f.body = cls._lower_block(f.body)
        return files

    @classmethod
    def _lower_block(cls, body: list[PyIRNode]) -> list[PyIRNode]:
        out: list[PyIRNode] = []
        for node in body:
            if isinstance(node, PyIRClassDef):
                out.extend(cls._lower_class(node))

            else:
                out.append(node)

        return out

    @classmethod
    def _lower_class(cls, node: PyIRClassDef) -> list[PyIRNode]:
        name = node.name
        parts = cls._collect_parts(node)

        out: list[PyIRNode] = []

        out.append(
            PyIRAssign(
                line=node.line,
                offset=node.offset,
                targets=[
                    PyIRVarCreate(
                        line=node.line,
                        offset=node.offset,
                        name=name,
                        is_global=True,
                    )
                ],
                value=PyIRDict(
                    line=node.line,
                    offset=node.offset,
                    items=[],
                ),
            )
        )

        out.append(
            PyIRAssign(
                line=node.line,
                offset=node.offset,
                targets=[
                    PyIRAttribute(
                        line=node.line,
                        offset=node.offset,
                        value=PyIRVarUse(
                            line=node.line,
                            offset=node.offset,
                            name=name,
                        ),
                        attr="__index",
                    )
                ],
                value=PyIRVarUse(
                    line=node.line,
                    offset=node.offset,
                    name=name,
                ),
            )
        )

        for assign in parts.fields:
            field_target = assign.targets[0]
            field_name = getattr(field_target, "name", None)

            if not isinstance(field_name, str) or not field_name:
                cls._die(
                    assign,
                    f"Некорректная цель присваивания поля класса {name}: ожидается имя переменной",
                )
                raise AssertionError("unreachable")

            out.append(
                PyIRAssign(
                    line=assign.line,
                    offset=assign.offset,
                    targets=[
                        PyIRAttribute(
                            line=assign.line,
                            offset=assign.offset,
                            value=PyIRVarUse(
                                line=assign.line,
                                offset=assign.offset,
                                name=name,
                            ),
                            attr=field_name,
                        )
                    ],
                    value=assign.value,
                )
            )

        for method in parts.methods:
            out.append(cls._lower_method(name, method))

        return out

    @classmethod
    def _collect_parts(cls, node: PyIRClassDef) -> _LoweredClassParts:
        raw_fields: list[PyIRAssign] = []
        methods: list[_LoweredMethod] = []

        for item in node.body:
            if isinstance(item, PyIRAssign):
                raw_fields.append(item)

            elif isinstance(item, PyIRFunctionDef):
                methods.append(cls._classify_method(item, node.name))

            else:
                cls._die(
                    item,
                    f"Недопустимая конструкция в теле класса {node.name}: {type(item).__name__}",
                )
                raise AssertionError("unreachable")

        fields: list[PyIRAssign] = []
        for a in raw_fields:
            fields.extend(cls._normalize_class_assign(a, node.name))

        for a in fields:
            cls._validate_class_assign(a, node.name)

        field_names: set[str] = set()
        for a in fields:
            t = a.targets[0]
            n = getattr(t, "name", None)
            if isinstance(n, str) and n:
                field_names.add(n)

        qualified_fields: list[PyIRAssign] = []
        for a in fields:
            qualified_fields.append(
                PyIRAssign(
                    line=a.line,
                    offset=a.offset,
                    targets=a.targets,
                    value=cls._qualify_class_locals(a.value, node.name, field_names),
                )
            )

        return _LoweredClassParts(qualified_fields, methods, field_names)

    @classmethod
    def _normalize_class_assign(
        cls, a: PyIRAssign, class_name: str
    ) -> list[PyIRAssign]:
        if len(a.targets) == 1:
            return [a]

        targets = a.targets
        value = a.value

        elements: list[PyIRNode] | None = None
        if isinstance(value, PyIRTuple):
            elements = list(value.elements)
        elif isinstance(value, PyIRList):
            elements = list(value.elements)

        if elements is None:
            cls._die(
                a,
                f"Некорректное поле класса {class_name}: допускается только одна цель присваивания (name = expr). "
                f"Распаковка разрешена только из literal tuple/list.",
            )
            raise AssertionError("unreachable")

        if len(elements) != len(targets):
            cls._die(
                a,
                f"Некорректное поле класса {class_name}: количество целей ({len(targets)}) "
                f"не совпадает с количеством значений ({len(elements)})",
            )
            raise AssertionError("unreachable")

        out: list[PyIRAssign] = []
        for t, v in zip(targets, elements, strict=True):
            name = getattr(t, "name", None)
            if not isinstance(name, str) or not name:
                cls._die(
                    a,
                    f"Некорректная цель распаковки в поле класса {class_name}: ожидается имя переменной",
                )
                raise AssertionError("unreachable")

            out.append(
                PyIRAssign(
                    line=a.line,
                    offset=a.offset,
                    targets=[t],
                    value=v,
                )
            )

        return out

    @classmethod
    def _validate_class_assign(cls, item: PyIRAssign, class_name: str) -> None:
        if len(item.targets) != 1:
            cls._die(
                item,
                f"Некорректное поле класса {class_name}: допускается только одна цель присваивания (name = expr)",
            )
            raise AssertionError("unreachable")

        target = item.targets[0]
        field_name = getattr(target, "name", None)

        if not isinstance(field_name, str) or not field_name:
            cls._die(
                item,
                f"Некорректная цель присваивания поля класса {class_name}: ожидается имя переменной",
            )
            raise AssertionError("unreachable")

        if not isinstance(target, (PyIRVarUse, PyIRVarCreate)):
            cls._die(
                item,
                f"Некорректное поле класса {class_name}: цель должна быть именем переменной",
            )
            raise AssertionError("unreachable")

    @classmethod
    def _classify_method(cls, fn: PyIRFunctionDef, class_name: str) -> _LoweredMethod:
        decorators = fn.decorators

        found_kind: _MethodKind | None = None
        found_dec: PyIRDecorator | None = None

        for dec in decorators:
            if dec.name == "classmethod":
                if found_kind is not None:
                    cls._die(
                        dec,
                        f"Метод {class_name}.{fn.name} имеет несколько семантических декораторов",
                    )
                    raise AssertionError("unreachable")
                found_kind = "class"
                found_dec = dec

            elif dec.name == "staticmethod":
                if found_kind is not None:
                    cls._die(
                        dec,
                        f"Метод {class_name}.{fn.name} имеет несколько семантических декораторов",
                    )
                    raise AssertionError("unreachable")
                found_kind = "static"
                found_dec = dec

        if found_kind is None:
            cls._validate_receiver_arity(fn, class_name, kind="instance")
            return _LoweredMethod("instance", fn)

        if found_kind == "class":
            cls._validate_receiver_arity(fn, class_name, kind="class")

        if found_dec is not None:
            cls._strip_known_decorator(fn, found_dec)

        return _LoweredMethod(found_kind, fn)

    @staticmethod
    def _validate_receiver_arity(
        fn: PyIRFunctionDef, class_name: str, kind: _MethodKind
    ) -> None:
        if kind in ("instance", "class"):
            if not fn.signature or len(fn.signature) < 1:
                LowerClassTablePass._die(
                    fn,
                    f"Метод {class_name}.{fn.name} должен иметь хотя бы один аргумент (ресивер)",
                )
                raise AssertionError("unreachable")

    @staticmethod
    def _strip_known_decorator(fn: PyIRFunctionDef, dec: PyIRDecorator) -> None:
        fn.decorators = [d for d in fn.decorators if d is not dec]

    @staticmethod
    def _lower_method(class_name: str, method: _LoweredMethod) -> PyIRFunctionDef:
        fn = method.fn
        fn.name = f"{class_name}.{fn.name}"
        return fn

    @classmethod
    def _qualify_class_locals(
        cls, node: PyIRNode, class_name: str, field_names: set[str]
    ) -> PyIRNode:
        if isinstance(node, PyIRVarUse) and node.name in field_names:
            return PyIRAttribute(
                line=node.line,
                offset=node.offset,
                value=PyIRVarUse(
                    line=node.line,
                    offset=node.offset,
                    name=class_name,
                ),
                attr=node.name,
            )

        if isinstance(node, PyIRAttribute):
            return PyIRAttribute(
                line=node.line,
                offset=node.offset,
                value=cls._qualify_class_locals(node.value, class_name, field_names),
                attr=node.attr,
            )

        if isinstance(node, PyIRSubscript):
            return PyIRSubscript(
                line=node.line,
                offset=node.offset,
                value=cls._qualify_class_locals(node.value, class_name, field_names),
                index=cls._qualify_class_locals(node.index, class_name, field_names),
            )

        if isinstance(node, PyIRCall):
            return PyIRCall(
                line=node.line,
                offset=node.offset,
                func=cls._qualify_class_locals(node.func, class_name, field_names),
                args_p=[
                    cls._qualify_class_locals(x, class_name, field_names)
                    for x in node.args_p
                ],
                args_kw={
                    k: cls._qualify_class_locals(v, class_name, field_names)
                    for k, v in node.args_kw.items()
                },
            )

        if isinstance(node, PyIRBinOP):
            return PyIRBinOP(
                line=node.line,
                offset=node.offset,
                op=node.op,
                left=cls._qualify_class_locals(node.left, class_name, field_names),
                right=cls._qualify_class_locals(node.right, class_name, field_names),
            )

        if isinstance(node, PyIRUnaryOP):
            return PyIRUnaryOP(
                line=node.line,
                offset=node.offset,
                op=node.op,
                value=cls._qualify_class_locals(node.value, class_name, field_names),
            )

        if isinstance(node, PyIRTuple):
            return PyIRTuple(
                line=node.line,
                offset=node.offset,
                elements=[
                    cls._qualify_class_locals(x, class_name, field_names)
                    for x in node.elements
                ],
            )

        if isinstance(node, PyIRList):
            return PyIRList(
                line=node.line,
                offset=node.offset,
                elements=[
                    cls._qualify_class_locals(x, class_name, field_names)
                    for x in node.elements
                ],
            )

        if isinstance(node, PyIRSet):
            return PyIRSet(
                line=node.line,
                offset=node.offset,
                elements=[
                    cls._qualify_class_locals(x, class_name, field_names)
                    for x in node.elements
                ],
            )

        if isinstance(node, PyIRDict):
            new_items: list[PyIRDictItem] = []
            for it in node.items:
                new_items.append(
                    PyIRDictItem(
                        line=it.line,
                        offset=it.offset,
                        key=cls._qualify_class_locals(it.key, class_name, field_names),
                        value=cls._qualify_class_locals(
                            it.value, class_name, field_names
                        ),
                    )
                )

            return PyIRDict(
                line=node.line,
                offset=node.offset,
                items=new_items,
            )

        if isinstance(node, PyIRFString):
            new_parts: list[str | PyIRNode] = []
            for p in node.parts:
                if isinstance(p, str):
                    new_parts.append(p)

                else:
                    new_parts.append(
                        cls._qualify_class_locals(p, class_name, field_names)
                    )

            return PyIRFString(
                line=node.line,
                offset=node.offset,
                parts=new_parts,
            )

        return node

    @staticmethod
    def _pos(node: PyIRNode | None) -> str:
        if node is None:
            return ""

        if node.line is not None and node.offset is not None:
            return f"\nLINE|OFFSET: {node.line}|{node.offset}"

        if node.line is not None:
            return f"\nLINE: {node.line}"

        return ""

    @classmethod
    def _die(cls, node: PyIRNode | None, msg: str) -> None:
        exit_with_code(1, f"{msg}{cls._pos(node)}")

from __future__ import annotations

from dataclasses import fields, is_dataclass

from ....py.ir_dataclass import PyIRComment, PyIRFile, PyIRImport, PyIRNode


class StripCommentsImportsPass:
    @staticmethod
    def run(ir: PyIRFile, *args) -> PyIRFile:
        def rw(node: PyIRNode) -> PyIRNode | None:
            if isinstance(node, PyIRComment):
                return None

            if not is_dataclass(node):
                return node

            for f in fields(node):
                v = getattr(node, f.name)

                if isinstance(v, PyIRNode):
                    nv = rw(v)
                    if nv is not None and nv is not v:
                        setattr(node, f.name, nv)

                elif isinstance(v, list):
                    changed = False
                    out_list: list[object] = []
                    for x in v:
                        if isinstance(x, PyIRNode):
                            nx = rw(x)
                            if nx is None:
                                changed = True
                                continue
                            changed = changed or (nx is not x)
                            out_list.append(nx)
                        else:
                            out_list.append(x)

                    if changed:
                        setattr(node, f.name, out_list)

                elif isinstance(v, dict):
                    changed = False
                    out_dict: dict[object, object] = {}
                    for k, x in v.items():
                        nk = k
                        nx = x

                        if isinstance(k, PyIRNode):
                            kk = rw(k)
                            if kk is None:
                                kk = k

                            nk = kk

                        if isinstance(x, PyIRNode):
                            xx = rw(x)
                            if xx is None:
                                changed = True
                                continue

                            nx = xx

                        changed = changed or (nk is not k) or (nx is not x)
                        out_dict[nk] = nx

                    if changed:
                        setattr(node, f.name, out_dict)

            return node

        new_body: list[PyIRNode] = []
        for n in ir.body:
            if isinstance(n, (PyIRComment, PyIRImport)):
                continue

            nn = rw(n)
            if nn is not None:
                new_body.append(nn)

        ir.body = new_body
        return ir

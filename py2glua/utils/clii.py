"""
clii

The easiest damned argparse wrapper there ever was.

Copyright 2020 James O'Beirne

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import functools
import inspect
import sys
import types
import typing as t
from textwrap import dedent

from .config import Config
from .logger_init import setup_logging

__VERSION__ = "1.0.3"


# Storage to that maps functions to a map of overrides for arguments. Allows
# users to manually specify ArgumentParser.add_arguments() arguments using
# the @cli.arg(...) decorator.
ARG_OVERRIDES_MAP: t.Dict[t.Callable, dict] = {}


class Arg:
    def __init__(
        self,
        param_name: str,
        flags: t.Sequence[str],
        type: object = str,
        help: str = "",
        default: object = inspect.Parameter.empty,
        is_kwarg: bool = False,
        is_vararg: bool = False,
        argparse_kwarg_overrides: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        self.param_name = param_name
        self.flags = flags
        # Store original parameter name unmangled (e.g. no '-' for '_' sub).

        if is_kwarg:
            self.flags = [n.replace("_", "-") for n in flags]

        self.type = type
        self.default = default
        self.is_kwarg = is_kwarg
        self.is_vararg = is_vararg
        self.help = help
        self.argparse_kwarg_overrides = argparse_kwarg_overrides or {}

    @classmethod
    def from_parameter(
        cls,
        param: inspect.Parameter,
        help: str = "",
        addl_options=None,
    ) -> "Arg":
        flags = []
        kwarg_overrides = None
        if addl_options:
            addl_flags, kwarg_overrides = addl_options
            for flag in addl_flags:
                flags.append(flag)

        flag = param.name.replace("_", "-")
        is_kwarg = param.default != inspect.Parameter.empty
        is_vararg = param.kind == inspect.Parameter.VAR_POSITIONAL

        if is_kwarg and not any(f.startswith("--") for f in flags):
            flags.append(f"--{flag}")

        elif not flags:
            flags = [param.name]

        return cls(
            param.name,
            flags,
            type=_get_callable_type(param.annotation),
            default=param.default,
            help=help,
            is_kwarg=is_kwarg,
            is_vararg=is_vararg,
            argparse_kwarg_overrides=kwarg_overrides,
        )

    @classmethod
    def from_func(cls, func: t.Callable) -> t.Sequence["Arg"]:
        # Ignore `**kwargs`; it can't be sensibly interpreted into flags
        params = [
            p for p in _get_func_params(func) if p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        addl_options = ARG_OVERRIDES_MAP.get(func, {})
        helps_from_doc = _get_helps_from_func(func, [p.name for p in params])

        return tuple(
            cls.from_parameter(
                param,
                helps_from_doc.get(param.name, ""),
                addl_options=addl_options.get(param.name),
            )
            for param in _get_func_params(func)
            if
            # Ignore `**kwargs`; it can't be sensibly interpreted into flags
            param.kind != inspect.Parameter.VAR_KEYWORD
        )

    def add_to_parser(self, parser: argparse.ArgumentParser):
        kwargs = dict(default=self.default, type=self.type, help=self.arg_help)

        disallowed_overrides = set(self.argparse_kwarg_overrides.keys()) & {
            "dest",
            "nargs",
        }
        if disallowed_overrides:
            raise ValueError(
                f"can't override add_argument() kwargs {disallowed_overrides}"
            )

        kwargs.update(self.argparse_kwarg_overrides)

        if self.is_vararg:
            kwargs["nargs"] = "*"
            kwargs.pop("default", "")
            if kwargs.get("type") == inspect.Parameter.empty:
                kwargs.pop("type")

        elif self.is_kwarg:
            kwargs["dest"] = self.param_name

        if self.type is bool and self.default in (True, False):
            kwargs["action"] = "store_false" if self.default else "store_true"
            kwargs.pop("type", "")

        parser.add_argument(*self.flags, **kwargs)  # type: ignore

    @property
    def arg_help(self) -> str:
        out = self.help or ""
        if self.default is not inspect.Parameter.empty:
            if out:
                out += ". "

            out += f"default: {self.default}"

        return out


def _get_func_params(func) -> t.List[inspect.Parameter]:
    return list(inspect.signature(func).parameters.values())


def _get_helps_from_func(func, param_names):
    if not func.__doc__:
        return {}

    helps = {}
    for line in dedent(func.__doc__).splitlines():
        stripped = line.lstrip()
        for p in param_names:
            if stripped.startswith(f"{p}:"):
                helps[p] = stripped[len(p) + 1 :].strip()
                break

    return helps


def _get_callable_type(annotation):
    if annotation is inspect.Parameter.empty:
        return str

    origin = t.get_origin(annotation)
    if origin is t.Union or origin is types.UnionType:
        args = t.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]

        if len(non_none) == 0:
            return str

        raise ValueError(
            f"Слишком много типов в Union: {annotation}. "
            f"argparse поддерживает только один тип преобразования."
        )

    if annotation is type(None):
        return str

    return annotation


class App:
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.subparsers = None
        self.args = argparse.Namespace()

    def add_arg(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
        return self.parser

    add_argument = add_arg

    def main(self, fnc):
        self.parser.set_defaults(func=fnc)

        for arg in Arg.from_func(fnc):
            arg.add_to_parser(self.parser)

        if not self.parser.description:
            self.parser.description = fnc.__doc__

        @functools.wraps(fnc)
        def wrapper(*args, **kwargs):
            return fnc(*args, **kwargs)

        return wrapper

    def cmd(self, fnc) -> t.Callable:
        if not self.subparsers:
            self.subparsers = self.parser.add_subparsers()

        desc = fnc.__doc__ or ""
        doclines = []

        for line in desc.splitlines():
            if line.strip().lower() in ["args:", "kwargs:"]:
                break
            doclines.append(line)

        sub = self.subparsers.add_parser(
            fnc.__name__.replace("_", "-"),
            description="\n".join(doclines),
        )

        for arg in Arg.from_func(fnc):
            arg.add_to_parser(sub)

        sub.set_defaults(func=fnc)
        return fnc

    def arg(self, name: str, *args, **kwargs) -> t.Callable:
        """
        Add additional ArgumentParser.add_argument() args for a certain arg.
        """

        def wrapper(fnc):
            ARG_OVERRIDES_MAP.setdefault(fnc, {})
            ARG_OVERRIDES_MAP[fnc][name] = (args, kwargs)
            return fnc

        return wrapper

    def parse_for_run(self) -> t.Tuple[t.Callable, t.Tuple[t.List, t.Dict]]:
        self.args = self.parser.parse_args()
        args = vars(self.args)
        fnc = args.pop("func", None)

        if not fnc:
            self.parser.print_help()
            sys.exit(1)

        func_args = []
        func_kwargs = {}
        building_kwargs = False

        # Only pull in those parameters which `fnc` accepts, since the
        # global parser may have supplied more.
        for p in _get_func_params(fnc):
            if p.kind == inspect.Parameter.KEYWORD_ONLY:
                building_kwargs = True

            if building_kwargs:
                func_kwargs[p.name] = args[p.name]

            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                func_args.extend(args[p.name])

            else:
                func_args.append(args[p.name])

        return (fnc, (func_args, func_kwargs))

    def run(self) -> object:
        (fnc, (func_args, func_kwargs)) = self.parse_for_run()

        setup_logging(self.args.debug)
        Config.cli_setup(self.args.debug, self.args.verbose)

        return fnc(*func_args, **func_kwargs)

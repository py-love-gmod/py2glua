from pathlib import Path

from ..glua import CompilerDirective


@CompilerDirective.gmod_api("AddCSLuaFile")
def AddCSLuaFile(file_path: Path) -> None: ...


@CompilerDirective.gmod_api("require")
def LuaRequire(file_path: Path) -> None: ...

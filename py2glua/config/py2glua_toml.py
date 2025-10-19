import tomllib
from pathlib import Path


class Py2GluaConfig:
    data = {}

    @classmethod
    def load(cls, path: Path) -> None:
        if not path.exists():
            raise RuntimeError("py2glua.toml is missing")

        dt = tomllib.loads(path.read_text("utf-8-sig"))

        if "system" not in dt:
            raise RuntimeError("Missing segment system in py2glua.toml")

        dt_sys = dt.pop("system")
        config_ver = dt_sys.get("version", None)
        if config_ver is None:
            raise ValueError("Missing system.version in py2glua.toml")

        func = getattr(cls, f"_v{config_ver}", None)
        if func is None or not callable(func):
            raise ValueError("Unknown config version in py2glua.toml")

        func(dt)

    @classmethod
    def _v1(cls, data: dict) -> None:
        # region base path
        build_data = data.get("build", {})
        source = Path(build_data.get("source", "./source"))
        output = Path(build_data.get("output", "./build"))

        data["build"]["source"] = source
        data["build"]["output"] = output
        # endregion

        cls.data = data

    @classmethod
    def default(cls) -> None:
        cls.data.clear()
        cls.data["build"] = {}

        cls.data["build"]["source"] = Path("./source")
        cls.data["build"]["output"] = Path("./build")
        cls.data["build"]["debug"] = False
        cls.data["build"]["clean_build"] = True
        cls.data["build"]["optimization"] = 0

import tempfile
from pathlib import Path

from py2glua.config import Py2GluaConfig
from py2glua.lang.py_ir_bulder import PyIRBuilder

with tempfile.TemporaryDirectory() as tmpdir:
    f = Path(tmpdir) / "temp_test.py"
    Py2GluaConfig.default()
    Py2GluaConfig.data["source"] = Path("/home/cain/Рабочий стол/py2glua/")
    f.write_text(
        """
import sys as s
from py2glua.config import Py2GluaConfig as cfg
del x
return x
""",
        encoding="utf-8",
    )

    ir = PyIRBuilder.build(f)
    print(ir)

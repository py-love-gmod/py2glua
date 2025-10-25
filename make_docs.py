import logging
import subprocess
import sys

modules = [
    "py2glua.glua",
]
output_dir = "docs"

cmd = ["pdoc"] + modules + ["-o", output_dir, "-d", "markdown"]
try:
    subprocess.run(
        [sys.executable, "-m", "pdoc"] + modules + ["-o", output_dir, "-d", "markdown"],
        check=True,
    )

except subprocess.CalledProcessError as err:
    logging.error(f"pdoc failed with exit code {err.returncode}")
    sys.exit(err.returncode)

except Exception as err:
    logging.exception("Unexpected error during docs build", err, stack_info=True)
    sys.exit(1)

sys.exit(0)

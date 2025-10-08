import logging
import sys
from importlib.metadata import PackageNotFoundError, version

from colorama import Fore, Style, init

init(autoreset=True)


class ColorFormater(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record) -> str:
        level_name = record.levelname
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{level_name}{Style.RESET_ALL}"
        return super().format(record)


logger = logging.getLogger("py2glua")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(ColorFormater("[%(levelname)s]: %(message)s"))
logger.addHandler(handler)


def _safe_version() -> str:
    try:
        return version("py2glua")

    except PackageNotFoundError:
        return "0.0.0-dev"


def main() -> None:
    logger.debug("py2glua")
    logger.debug(f"Version: {_safe_version()}")

    sys.exit(0)


if __name__ == "__main__":
    main()

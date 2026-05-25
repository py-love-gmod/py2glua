from compiler import Compiler
from plg_reader import build_python_files_dir
from utils import Config, Shutdown, logger


def build_cmd() -> None:
    # По факту метод занимается онли санити проверками.
    # Можно было бы сделать это более адекватно? Да.
    # Мне похуй? Абсолютно
    logger.debug(Config._data)

    in_p = Config.get_path_input()
    if not in_p.exists():
        Shutdown.user_error(f"Указанного пути не существует: {in_p.absolute()}")

    if not in_p.is_dir():
        Shutdown.user_error(
            f"Указанный путь к дерриктории с кодом не является папкой: {in_p.absolute()}"
        )

    try:
        irs = build_python_files_dir(Config.get_path_input())



    except Exception as e:
        Shutdown.user_error(str(e))

    Compiler.build(irs)

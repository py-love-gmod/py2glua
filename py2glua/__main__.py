from pathlib import Path

from _compiler import Compiler
from plg_reader import clii
from _utils import Config, Shutdown, init_log_config

cli = clii.App(description=__doc__)


@cli.cmd
def version() -> None:
    """Напечатать версию утилиты"""
    print(Config.version(), end="")


@cli.cmd
@cli.arg("input_path", "-i")
@cli.arg("output_path", "-o")
@cli.arg("plg_path", "-p")
@cli.arg("namespace", "-n")
def build(
    input_path: Path = Path("./source/"),
    output_path: Path = Path("./output/"),
    plg_path: Path = Path("./.plg/"),
    namespace: str | None = None,
) -> None:
    """Начать компиляцию проекта

    Args:
        input_path: Дерриктория с py кодом
        output_path: Дерриктория с lua выходом
        plg_path: Дерриктория plg
        namespace: Неймспейс аддона
    """
    if not input_path.exists():
        Shutdown.user_error(f"Указанного пути не существует: {input_path.absolute()}")

    if not input_path.is_dir():
        Shutdown.user_error(
            f"Указанный путь к дерриктории с кодом не является папкой: {input_path.absolute()}"
        )

    Config.path_setup(input_path, output_path, plg_path)
    Config.namespace_setup(namespace)
    Compiler.compile()


if __name__ == "__main__":
    try:
        cli.run(init_log_config)

    except KeyboardInterrupt:
        Shutdown.normal("Работа программы прервано пользователем")

    except Exception as e:
        Shutdown.softwear_error(f"Произошла ошибка компилятора: {e}")

from pathlib import Path

from cli import build_cmd, init_cmd
from clii.clii import App
from utils import Config, setup_logging

cli = App(description=__doc__)
cli.add_arg("--verbose", "-v", action="store_true", default=False)
cli.add_arg("--debug", "-d", action="store_true", default=False)


@cli.cmd
def version() -> None:
    """Версия утилиты"""
    print(Config.version(), end="")


@cli.cmd
@cli.arg("input_path", "-i")
@cli.arg("output_path", "-o")
def init(
    input_path: Path = Path("./source"),
    output_path: Path = Path("./output"),
) -> None:
    """Инициализация папок компилятора

    Args:
        input_path: Дерриктория с py кодом

        output_path: Дерриктория с lua выходом
    """
    init_cmd(input_path, output_path, Path(".plg"))


@cli.cmd
def build() -> None:
    """Начать компиляцию проекта"""
    build_cmd()


if __name__ == "__main__":
    cli.run()
    Config.cli_args = cli.args
    setup_logging(cli.args.debug)

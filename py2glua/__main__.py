from pathlib import Path

from cli import build_cmd
from utils import App, Config, Shutdown

cli = App(description=__doc__)
cli.add_arg(
    "--verbose",
    "-v",
    help="Более болтливый компилятор",
    action="store_true",
    default=False,
)
cli.add_arg(
    "--debug",
    "-d",
    help="Отладочный вывод",
    action="store_true",
    default=False,
)


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
    Config.path_setup(input_path, output_path, plg_path)
    Config.namespace_setup(namespace)

    build_cmd()


if __name__ == "__main__":
    try:
        cli.run()

    except KeyboardInterrupt:
        Shutdown.normal("Прервано пользователем")

    except Exception as e:
        Shutdown.softwear_error(f"Произошла ошибка компилятора: {e}")

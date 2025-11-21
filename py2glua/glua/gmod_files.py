from pathlib import Path


class GmodBasePath:
    """Все пути, используемые gmod."""

    CODE_FILE_DIR: Path
    """Путь к исходному коду ТЕКУЩЕГО аддона"""

    DATA_FILE_DIR: Path
    """Путь к рабочей директории данных (gmod data folder)"""

    CUR_FILE: Path = Path(".")
    """Текущий исполняемый файл"""

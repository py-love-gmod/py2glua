import sys
from pathlib import Path

from generator import FunctionFilter, TypeOverride, generate_all
from reader import load_objects_from_zip
from schema_builder import build_full_schema

DATA_DIR = Path(__file__).parents[1] / "data"

TYPE_OVERRIDES_PATH = DATA_DIR / "type_overrides.txt"
NUKE_PATH = DATA_DIR / "nuke.txt"


def find_zip() -> Path:
    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        print(f"В папке {DATA_DIR} нет .zip файлов.")
        sys.exit(1)

    if len(zip_files) > 1:
        print(f"В папке {DATA_DIR} больше 1 .zip файла")
        sys.exit(1)

    return zip_files[0]


def main() -> None:
    zip_path = find_zip()

    print(f"Чтение {zip_path}...")
    entries = load_objects_from_zip(zip_path)
    print(f"Загружено файлов: {len(entries)}")

    type_override = TypeOverride(TYPE_OVERRIDES_PATH)
    function_filter = FunctionFilter(NUKE_PATH)

    output = Path("generated")
    output.mkdir(exist_ok=True)

    generate_all(build_full_schema(entries), output, type_override, function_filter)
    print("Готово.")


if __name__ == "__main__":
    main()

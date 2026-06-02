import sys
from pathlib import Path

from dispatch import generate_all
from generator.filter import FunctionFilter
from generator.type_override import TypeOverride
from reader import load_objects_from_zip
from schema_builder import build_full_schema

DATA_DIR = Path(__file__).parents[1] / "data"
TypeOverride.init(DATA_DIR)
FunctionFilter.init(DATA_DIR)


def find_zip() -> Path:
    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        print(f"В папке {DATA_DIR} нет .zip файлов.")
        sys.exit(1)

    if len(zip_files) > 1:
        print(f"В папке {DATA_DIR} больше 1 .zip файла")
        sys.exit(1)

    return zip_files[0]


def main():
    zip_path = find_zip()

    print(f"Чтение {zip_path}...")
    entries = load_objects_from_zip(zip_path)
    print(f"Загружено файлов: {len(entries)}")

    output = Path("generated")
    output.mkdir(exist_ok=True)
    generate_all(build_full_schema(entries), output)
    print("Готово.")


if __name__ == "__main__":
    main()

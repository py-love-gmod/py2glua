import sys
from pathlib import Path

from dispatch import generate_all
from reader import load_objects_from_zip
from schema_builder import build_full_schema

SELECTED_CATEGORIES = {}


def find_zip(directory: Path) -> Path:
    zip_files = list(directory.glob("*.zip"))
    if not zip_files:
        print(f"В папке {directory} нет .zip файлов.")
        sys.exit(1)

    return zip_files[0]


def main():
    script_dir = Path(__file__).parents[1]
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
        if not zip_path.exists():
            print(f"Архив не найден: {zip_path}")
            sys.exit(1)

    else:
        zip_path = find_zip(script_dir)

    print(f"Чтение {zip_path}...")
    entries = load_objects_from_zip(zip_path)
    print(f"Загружено файлов: {len(entries)}")

    schema = build_full_schema(entries)

    if SELECTED_CATEGORIES:
        filtered = {k: v for k, v in schema.items() if k in SELECTED_CATEGORIES}
        print(f"Оставлены категории: {list(filtered.keys())}")
    else:
        filtered = schema

    output = Path("generated")
    output.mkdir(exist_ok=True)
    generate_all(filtered, output)
    print("Готово.")


if __name__ == "__main__":
    main()

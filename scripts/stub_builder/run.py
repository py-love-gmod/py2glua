import sys
from pathlib import Path

from dispatch import generate_all
from reader import load_objects_from_zip
from schema_builder import build_full_schema


def find_zip() -> Path:
    script_dir = Path(__file__).parents[1] / "data"
    zip_files = list(script_dir.glob("*.zip"))
    if not zip_files:
        print(f"В папке {script_dir} нет .zip файлов.")
        sys.exit(1)

    if len(zip_files) > 1:
        print(f"В папке {script_dir} больше 1 .zip файла")
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

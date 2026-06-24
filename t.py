import os

OUTPUT_FILE = "combined.py"

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk(r"F:\Desktop\plg\py2glua\py2glua"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                out.write(f"\n\n# File: {path}\n\n")
                with open(path, "r", encoding="utf-8") as inp:
                    out.write(inp.read())

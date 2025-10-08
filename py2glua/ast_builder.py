import ast
import logging
from pathlib import Path

logger = logging.getLogger("py2glua")

SYN_ERR_MSG = """
[astb] Синтаксическая ошибка при предкомпиляции:
Path         : {path}
Line | Offset: {line_number} | {line_offset}
Msg          : {msg}
Source       : {source_line}
"""


class ASTBuilder:
    @staticmethod
    def load_from_file(path: Path) -> ast.Module | None:
        """Создаёт AST-модель из Python-файла.

        При ошибках пишет лог и возвращает None.

        Args:
            path (Path): путь к .py файлу

        Returns:
            ast.Module | None: дерево модуля или None при ошибке
        """
        if not path.exists():
            logger.warning(f"[astb] '{path}' не существует")
            return None

        if path.suffix.lower() != ".py":
            logger.warning(f"[astb] '{path}' не является Python-файлом")
            return None

        try:
            source = path.read_text(encoding="utf-8-sig")

        except UnicodeDecodeError as err:
            logger.error(f"[astb] Ошибка декодирования файла '{path}': {err}")
            return None

        except Exception as err:
            logger.error(f"[astb] Ошибка чтения файла '{path}': {err}")
            return None

        try:
            ast_struct = ast.parse(source, filename=str(path))

        except SyntaxError as err:
            source_line = (err.text or "").strip() if err.text else ""
            logger.error(
                SYN_ERR_MSG.format(
                    path=path,
                    line_number=err.lineno or "?",
                    line_offset=err.offset or "?",
                    msg=err.msg or "Неизвестная ошибка",
                    source_line=source_line,
                )
            )
            return None

        except Exception as err:
            logger.exception(f"[astb] Не удалось разобрать AST для '{path}': {err}")
            return None

        return ast_struct

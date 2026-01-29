import sys
from pathlib import Path

from .._lang.py.ir_builder import PyIRNode
from .logging_setup import logger


class CompilerExit:
    @classmethod
    def generic(
        cls,
        code: int,
        msg: str | None = None,
    ) -> None:
        if msg:
            if code == 0:
                logger.info(msg)

            else:
                logger.error(msg)
                logger.error(f"Exit code: {code}")

        sys.exit(code)

    @classmethod
    def user_error(
        cls,
        msg: str,
        path: Path | None = None,
        line: int | None = None,
        offset: int | None = None,
        *,
        show_path: bool = True,
        show_pos: bool = True,
    ) -> None:
        log_msg = msg
        if show_path:
            if path is None:
                log_msg += "\nФайл: неизвестно"
            else:
                log_msg += f"\nФайл: {path.resolve()}"

        if show_pos:
            log_msg += f"\nLINE|OFFSET: {line if line else 0}|{offset if offset else 0}"

        logger.error(log_msg)
        logger.error("Exit code: 1")
        sys.exit(1)

    @classmethod
    def user_error_node(
        cls,
        msg: str,
        path: Path | None,
        node: PyIRNode,
        *,
        show_path: bool = True,
        show_pos: bool = True,
    ) -> None:
        cls.user_error(
            msg,
            path,
            line=node.line,
            offset=node.offset,
            show_path=show_path,
            show_pos=show_pos,
        )

    @classmethod
    def _exit_with_log(
        cls,
        code: int,
        msg: str,
        *,
        critical: bool = False,
    ) -> None:
        if critical:
            logger.critical(msg)
        else:
            logger.error(msg)

        logger.error(f"Exit code: {code}")
        sys.exit(code)

    @classmethod
    def system_error(cls, msg: str, is_critical: bool = False) -> None:
        cls._exit_with_log(2, msg, critical=is_critical)

    @classmethod
    def internal_error(cls, msg: str, is_critical: bool = False) -> None:
        cls._exit_with_log(3, msg, critical=is_critical)

r"""Contain utility functions to configure the standard logging
library."""

from __future__ import annotations

__all__ = ["configure_logging"]

import logging

from grizz.utils.imports import is_colorlog_available

if is_colorlog_available():  # pragma: no cover
    import colorlog

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    r"""Configure the logging module with a colored formatter.

    Args:
        level: The lower level.
    """
    if not is_colorlog_available():
        logging.basicConfig(level=level)
        return

    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        fmt=(
            "%(log_color)s(%(process)d) %(asctime)s [%(levelname)s] %(name)s:%(lineno)s%(reset)s "
            "%(message_log_color)s%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "bold_yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={
            "message": {
                "DEBUG": "cyan",
                "INFO": "reset",
                "WARNING": "bold_yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        },
    )
    handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[handler])

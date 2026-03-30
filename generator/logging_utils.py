"""
Shared file-backed logging utilities for generator workflows.
"""

import logging
from pathlib import Path


RUN_LOGGER_NAME = "generator.run"


def _get_base_logger() -> logging.Logger:
    """Return the base run logger with a silent default handler."""
    logger = logging.getLogger(RUN_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def get_run_logger(module_name: str | None = None) -> logging.Logger:
    """Return a module-scoped logger under the generator run logger."""
    _get_base_logger()
    if module_name:
        return logging.getLogger(f"{RUN_LOGGER_NAME}.{module_name}")
    return logging.getLogger(RUN_LOGGER_NAME)


def configure_run_logger(log_path: Path) -> logging.Logger:
    """Configure the generator run logger to write to the given log file."""
    logger = _get_base_logger()
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    return logger

"""Унифицированный логгер для всех модулей проекта."""
import logging
import sys
from pathlib import Path
from src.config import LOGS_DIR


def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(LOGS_DIR / log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
"""Structured logging setup. Single entry point: `setup_logging()`."""

from __future__ import annotations

import logging
import sys
from logging import Logger

from app.config import settings


_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging() -> None:
    """Configure root logger once; safe to call multiple times."""
    root = logging.getLogger()
    if getattr(root, "_configured", False):
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root.handlers = [handler]
    # Tame noisy libs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    root._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> Logger:
    return logging.getLogger(name)

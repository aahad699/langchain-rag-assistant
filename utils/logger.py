"""Centralised, dependency-free logging for the RAG application."""

import logging
import os
import sys


def get_logger(name: str = "rag") -> logging.Logger:
    """
    Return a configured logger.

    The log level is controlled by the ``LOG_LEVEL`` environment variable
    (defaults to ``INFO``). Handlers are attached only once per logger so
    repeated calls are safe.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(handler)
    logger.propagate = False

    return logger

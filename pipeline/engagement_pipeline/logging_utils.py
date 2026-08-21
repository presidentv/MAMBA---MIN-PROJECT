"""Console + rotating-file logging shared by both phase drivers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger once and return the pipeline logger.

    Repeated calls replace existing handlers so that re-running a phase inside the
    same interpreter (e.g. from a notebook) does not duplicate every line.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # MediaPipe / absl are extremely chatty at INFO on every inference call.
    logging.getLogger("absl").setLevel(logging.ERROR)
    return logging.getLogger("pipeline")

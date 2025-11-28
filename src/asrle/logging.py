from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from rich.console import Console
from rich.logging import RichHandler


@dataclass(frozen=True)
class LogConfig:
    level: str = os.getenv("ASRLE_LOG_LEVEL", "INFO")


def setup_logging(cfg: LogConfig | None = None) -> logging.Logger:
    cfg = cfg or LogConfig()
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True)],
    )
    logger = logging.getLogger("asrle")
    logger.setLevel(level)
    return logger

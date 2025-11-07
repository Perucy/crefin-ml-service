"""
    Logging Configuration
    Structured logging for debugging and monitoring
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from config.settings import settings

def setup_logger(name: str = "ml_service") -> logging.Logger:
    """
        Setup structured logger
    """

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level))

    # remove existing handlers
    logger.handlers.clear()

    # console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))

    # format: timestamp - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # file handler 
    if settings.is_production:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# global logger instance
logger = setup_logger()

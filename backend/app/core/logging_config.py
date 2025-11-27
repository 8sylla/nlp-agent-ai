import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(log_level: str = "INFO"):
    """Configure structured logging"""

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # Console handler with JSON format
    handler = logging.StreamHandler(sys.stdout)

    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        timestamp=True
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

logger = setup_logging()
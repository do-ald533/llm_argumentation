"""Logging configuration using structlog."""
import sys
import logging
from pathlib import Path
from datetime import datetime
import structlog


def setup_logging(logs_dir: Path = None, log_level: str = "INFO") -> structlog.BoundLogger:
    """
    Configure structlog for the application.
    
    Args:
        logs_dir: Directory to save log files (default: logs/)
        log_level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    if logs_dir is None:
        logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    
    logger = structlog.get_logger()
    logger.info("logging_initialized", log_file=str(log_file), level=log_level)
    
    return logger


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Optional logger name for context
    
    Returns:
        Logger instance
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(module=name)
    return logger

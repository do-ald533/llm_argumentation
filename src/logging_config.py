"""Logging configuration using structlog."""
import sys
import logging
from pathlib import Path
from datetime import datetime
import structlog

# Module-level debug logger (file-only, written once setup_logging() is called)
_debug_logger: logging.Logger = None


def setup_debug_file_logger(logs_dir: Path, timestamp: str) -> logging.Logger:
    """Create a plain file logger that writes to logs/debug_<timestamp>.log.

    This logger is separate from structlog and captures fine-grained debug
    traces (LLM responses, intermediate variables) for the pipeline tasks.
    """
    global _debug_logger
    debug_file = logs_dir / f"debug_{timestamp}.log"
    logger = logging.getLogger("pipeline_debug")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(debug_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False  # keep out of root / structlog handlers
    _debug_logger = logger
    logger.info(f"debug_logger_initialized log_file={debug_file}")
    return logger


def get_debug_logger() -> logging.Logger:
    """Return the shared debug file logger (lazy init if setup_logging not called)."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = logging.getLogger("pipeline_debug")
    return _debug_logger


def setup_logging(logs_dir: Path = None, log_level: str = "INFO") -> structlog.BoundLogger:
    """
    Configure structlog for the application.
    
    Args:
        logs_dir: Directory to save log files (default: logs/)
        log_level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    if logs_dir is None:
        logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    # Dedicated fine-grained debug log file
    setup_debug_file_logger(logs_dir, timestamp)
    
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

"""
FrameNotes Logging Module
Enhanced with debug mode, context prefixes, and beautified output.
"""

import logging
import sys
import os
import re
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Callable
import threading
from functools import wraps
from contextlib import contextmanager


class LogContext:
    """Predefined operation contexts for structured logging"""
    UI = "UI"
    TRANSCRIBE = "TRANSCRIBE"
    ANALYZE = "ANALYZE"
    CONFIG = "CONFIG"
    FILE = "FILE"
    API = "API"
    EXPORT = "EXPORT"
    INIT = "INIT"
    PROCESS = "PROCESS"


class LogSeparators:
    """Visual separators for major operations"""
    MAJOR = "=" * 60
    MINOR = "-" * 40
    START = ">>> START "
    END = "<<< END "
    STEP = ">>>"


class LogColors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DEBUG = "\033[36m"
    INFO = "\033[32m"
    WARNING = "\033[33m"
    ERROR = "\033[31m"
    CRITICAL = "\033[35m"
    TIMESTAMP = "\033[90m"
    MODULE = "\033[34m"
    CONTEXT = "\033[95m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for console output"""
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }
    # Pre-compile regex for context highlighting (performance optimization)
    CONTEXT_PATTERN = re.compile(r"\[([A-Z_]+)\]")
    def __init__(self, fmt, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and self._supports_color()

    def _supports_color(self):
        if sys.platform == "win32":
            return os.environ.get("TERM") or os.environ.get("WT_SESSION")
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def format(self, record):
        record = logging.makeLogRecord(record.__dict__)
        if self.use_colors:
            level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
            record.levelname = f"{level_color}{record.levelname:8}{LogColors.RESET}"
            record.name = f"{LogColors.MODULE}{record.name}{LogColors.RESET}"
            if "[" in str(record.msg) and "]" in str(record.msg):
                record.msg = self.CONTEXT_PATTERN.sub(f"{LogColors.CONTEXT}[\1]{LogColors.RESET}", str(record.msg))
        return super().format(record)


class BeautifiedFileFormatter(logging.Formatter):
    """Enhanced file formatter with milliseconds"""
    # Pre-compile regex for performance (avoid recompiling on every format call)
    ANSI_ESCAPE_PATTERN = re.compile(r"(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record):
        formatted = super().format(record)
        return self.ANSI_ESCAPE_PATTERN.sub("", formatted)

class FrameNotesLogger:
    """Thread-safe singleton logger"""
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    _debug_mode = False

    DEFAULT_LOG_DIR = "logs"
    DEFAULT_LOG_FILE = "framenotes.log"
    DEFAULT_MAX_BYTES = 10 * 1024 * 1024
    DEFAULT_BACKUP_COUNT = 5
    DEFAULT_CONSOLE_LEVEL = logging.INFO
    DEFAULT_FILE_LEVEL = logging.DEBUG

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir=None, log_file=None, console_level=None, file_level=None, max_bytes=None, backup_count=None, use_colors=True):
        if FrameNotesLogger._initialized:
            return
        with self._lock:
            if FrameNotesLogger._initialized:
                return
            self.log_dir = Path(log_dir or self.DEFAULT_LOG_DIR)
            self.log_file = log_file or self.DEFAULT_LOG_FILE
            self.console_level = console_level or self.DEFAULT_CONSOLE_LEVEL
            self.file_level = file_level or self.DEFAULT_FILE_LEVEL
            self.max_bytes = max_bytes or self.DEFAULT_MAX_BYTES
            self.backup_count = backup_count or self.DEFAULT_BACKUP_COUNT
            self.use_colors = use_colors
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._setup_logging()
            FrameNotesLogger._initialized = True

    def _setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S", use_colors=self.use_colors)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        log_path = self.log_dir / self.log_file
        file_handler = RotatingFileHandler(log_path, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8")
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(BeautifiedFileFormatter())
        root_logger.addHandler(file_handler)

        logging.getLogger("framenotes").info(f"[{LogContext.INIT}] Logging initialized - Console: {logging.getLevelName(self.console_level)}, File: {log_path}")

    @classmethod
    def get_logger(cls, name):
        if not cls._initialized:
            cls()
        return logging.getLogger(name)

    @classmethod
    def set_console_level(cls, level):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                handler.setLevel(level)

    @classmethod
    def set_file_level(cls, level):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                handler.setLevel(level)

    @classmethod
    def set_debug_mode(cls, enabled):
        with cls._lock:
            cls._debug_mode = enabled
        if enabled:
            cls.set_console_level(logging.DEBUG)
            logging.getLogger("framenotes").info(f"[{LogContext.CONFIG}] Debug mode ENABLED")
        else:
            cls.set_console_level(logging.INFO)
            logging.getLogger("framenotes").info(f"[{LogContext.CONFIG}] Debug mode DISABLED")

    @classmethod
    def is_debug_mode(cls):
        with cls._lock:
            return cls._debug_mode

    @classmethod
    def get_log_file_path(cls):
        with cls._lock:
            if cls._instance:
                return cls._instance.log_dir / cls._instance.log_file
            return None


def get_logger(name):
    return FrameNotesLogger.get_logger(name)


def init_logging(log_dir=None, console_level=logging.INFO, file_level=logging.DEBUG, verbose=False):
    if verbose:
        console_level = logging.DEBUG
    FrameNotesLogger(log_dir=log_dir, console_level=console_level, file_level=file_level)


def set_debug_mode(enabled):
    FrameNotesLogger.set_debug_mode(enabled)


def is_debug_mode():
    return FrameNotesLogger.is_debug_mode()


def get_log_file_path():
    return FrameNotesLogger.get_log_file_path()


def log_function(context=LogContext.PROCESS):
    """Decorator to log function entry/exit with timing"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.debug(f"[{context}] {LogSeparators.START}{func.__name__}()")
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.debug(f"[{context}] {LogSeparators.END}{func.__name__}() completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.error(f"[{context}] {LogSeparators.END}{func.__name__}() FAILED: {e}")
                raise
        return wrapper
    return decorator


@contextmanager
def log_operation(name, context=LogContext.PROCESS, logger=None):
    """Context manager for logging operations with timing"""
    if logger is None:
        logger = get_logger("framenotes")
    logger.info(f"[{context}] {LogSeparators.MAJOR}")
    logger.info(f"[{context}] {LogSeparators.START}{name}")
    start_time = datetime.now()
    try:
        yield
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{context}] {LogSeparators.END}{name} (completed in {elapsed:.2f}s)")
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"[{context}] {LogSeparators.END}{name} FAILED: {e}")
        raise
    finally:
        logger.info(f"[{context}] {LogSeparators.MAJOR}")


def log_step(step_num, total_steps, description, context=LogContext.PROCESS, logger=None):
    """Log a processing step with progress indicator"""
    if logger is None:
        logger = get_logger("framenotes")
    logger.info(f"[{context}] {LogSeparators.STEP} Step {step_num}/{total_steps}: {description}")


class ProcessingLogger:
    """Context manager for logging video processing operations"""
    def __init__(self, video_path, operation="processing", context=LogContext.PROCESS):
        self.video_path = Path(video_path).name
        self.operation = operation
        self.context = context
        self.logger = get_logger("framenotes.processing")
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"[{self.context}] {LogSeparators.MAJOR}")
        self.logger.info(f"[{self.context}] {LogSeparators.START}{self.operation}: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"[{self.context}] {LogSeparators.END}{self.operation}: {self.video_path} (completed in {duration.total_seconds():.2f}s)")
        else:
            self.logger.error(f"[{self.context}] {LogSeparators.END}{self.operation}: {self.video_path} FAILED: {exc_val}")
        self.logger.info(f"[{self.context}] {LogSeparators.MAJOR}")
        return False

    def progress(self, current, total, message=""):
        pct = (current / total * 100) if total > 0 else 0
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "=" * filled + "-" * (bar_width - filled)
        self.logger.debug(f"[{self.context}] [{bar}] {pct:.1f}% ({current}/{total}) {message}")

    def step(self, step_name, step_num=None, total_steps=None):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if step_num and total_steps:
            self.logger.info(f"[{self.context}] {LogSeparators.STEP} Step {step_num}/{total_steps}: {step_name} (elapsed: {elapsed:.2f}s)")
        else:
            self.logger.info(f"[{self.context}] {LogSeparators.STEP} {step_name} (elapsed: {elapsed:.2f}s)")


class APILogger:
    """Logger for API request/response debugging"""
    def __init__(self):
        self.logger = get_logger("framenotes.api")

    def log_request(self, endpoint, model, tokens_estimate=0):
        self.logger.debug(f"[{LogContext.API}] REQUEST: {endpoint} | Model: {model} | Est. tokens: {tokens_estimate}")

    def log_response(self, status, tokens_used=0, duration_ms=0):
        self.logger.debug(f"[{LogContext.API}] RESPONSE: {status} | Tokens: {tokens_used} | Duration: {duration_ms}ms")

    def log_error(self, error_type, message, retryable=False):
        retry_str = " (retryable)" if retryable else ""
        self.logger.error(f"[{LogContext.API}] ERROR{retry_str}: {error_type} - {message}")


api_logger = APILogger()


if __name__ == "__main__":
    init_logging(verbose=True)
    logger = get_logger(__name__)
    logger.debug(f"[{LogContext.INIT}] Debug message")
    logger.info(f"[{LogContext.INIT}] Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print(f"Log file: {get_log_file_path()}")

import sys
from loguru import logger # type: ignore
from typing import Optional


class LogFactory:
    _configured = False

    @classmethod
    def get_logger(cls, name: Optional[str] = None):
        """
        Returns a loguru logger instance.
        Configures the logger once with a standard console output.

        Args:
            name (Optional[str]): Context name (not used directly in loguru but could be passed as extra).

        Returns:
            loguru.Logger: Configured logger.
        """
        if not cls._configured:
            cls._configure_logger()
            cls._configured = True

        return logger.bind(module=name or "app")

    @classmethod
    def _configure_logger(cls):
        """
        Sets up loguru to output logs to stdout with standard formatting.
        """
        logger.remove()  # Remove default logger to avoid duplication

        logger.add(
            sys.stdout,
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{extra[module]}</cyan> | "
                   "<level>{message}</level>",
            enqueue=True,  # Thread-safe
            backtrace=True,  # Show full trace on error
            diagnose=True,  # Show variable values on exceptions
        )
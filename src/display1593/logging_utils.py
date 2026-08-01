"""Helper for entry-point scripts to set up logging.

Call configure_root_logging() once, near the top of a script that is
run directly (not imported as a library) - e.g. digclock.py or
schelling.py. It attaches a single, size-capped rotating file handler
to the *root* logger rather than to the script's own module logger.

Because Python loggers propagate log records up to the root logger by
default, that one handler then captures log records from every module
the script uses - display1593, display1593.lock, serial_comm, and the
script itself - in one file, without each of those modules needing to
set up its own logging.
"""

import logging
import logging.handlers
import os

DEFAULT_MAX_BYTES = 256 * 1024
DEFAULT_BACKUP_COUNT = 2
LOG_FORMAT = "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s"


def configure_root_logging(
    log_path,
    level=logging.INFO,
    max_bytes=DEFAULT_MAX_BYTES,
    backup_count=DEFAULT_BACKUP_COUNT,
):
    """Attach a size-capped RotatingFileHandler to the root logger.

    log_path: file to log to.
    level: minimum severity to record (see the `logging` module).
    max_bytes: the log file is rotated once it reaches this size, so
        it can't grow without bound and fill up the RPi's storage.
    backup_count: number of rotated (older) log files to keep, in
        addition to the current one.

    Safe to call more than once with the same log_path (e.g. if the
    script's module is imported more than once, such as during
    testing) - it will not add a duplicate handler.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    log_path = os.path.abspath(str(log_path))
    already_configured = any(
        getattr(handler, "baseFilename", None) == log_path
        for handler in root_logger.handlers
    )
    if already_configured:
        return

    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(handler)

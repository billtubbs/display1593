"""Utility for ensuring only one process controls the LED display at a
time.

Uses fcntl.flock(), a POSIX "advisory" file lock built into the OS: any
process can request a lock on a file, and the OS tracks who holds it.
It's called "advisory" because it only affects other processes that
also use flock() on the same file - it doesn't stop someone from just
reading or writing the file directly. That's fine for our purposes: the
lock file's contents are never used, it's only acted as a shared handle
that processes coordinate around.

The key property we rely on is that flock() can *block*: a process that
asks for a lock already held by another process will simply pause there
until it becomes available, rather than raising an error immediately.
And if the process holding the lock exits or crashes for any reason,
the OS releases the lock automatically (it's tied to the process's open
file descriptor), so there's no stale lock file to clean up by hand.
"""

import fcntl
import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_LOCK_PATH = "/tmp/display1593.lock"
DEFAULT_TIMEOUT = 5.0
POLL_INTERVAL = 0.2


class DisplayLockTimeout(Exception):
    """Raised when the lock is still held by another process after
    waiting for the given timeout."""


class DisplayLock:
    """An OS-level exclusive lock, with a wait-then-give-up timeout.

    Only one DisplayLock (in any process, on this machine) can be
    holding the lock at the same path at a time. Calling acquire()
    waits for up to `timeout` seconds for the lock to become free; if
    it's still held by another process after that, it raises
    DisplayLockTimeout rather than waiting forever. This is used to
    stop two scripts from talking to the display's microcontrollers
    over serial at the same time, which corrupts the communication
    between them.

    Can be used directly:

        lock = DisplayLock()
        lock.acquire()
        try:
            ...
        finally:
            lock.release()

    or as a context manager:

        with DisplayLock():
            ...
    """

    def __init__(self, path=DEFAULT_LOCK_PATH):
        self.path = path
        self._file = None

    def acquire(self, timeout=DEFAULT_TIMEOUT):
        """Try to acquire the lock, giving up after `timeout` seconds.

        Opens (creating if necessary) the lock file, then repeatedly
        asks fcntl.flock() for an exclusive, non-blocking lock
        ("LOCK_EX | LOCK_NB") on it, pausing POLL_INTERVAL seconds
        between attempts. flock() has no built-in timeout, so this
        polling loop is what turns it into a "wait up to N seconds"
        operation: each individual attempt returns immediately (either
        it succeeds, or raises BlockingIOError because another process
        still holds it), and we keep retrying until either it succeeds
        or `timeout` seconds have passed, at which point we raise
        DisplayLockTimeout instead of continuing to wait.

        Calling this again while already held (by this same
        DisplayLock instance) does nothing, so it's safe to call more
        than once.
        """
        if self._file is not None:
            return

        self._file = open(self.path, "w")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.debug("Display lock acquired.")
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._file.close()
                    self._file = None
                    raise DisplayLockTimeout(
                        f"Could not acquire display lock ({self.path}) "
                        f"within {timeout}s - another process is "
                        "likely controlling the display."
                    ) from None
                time.sleep(POLL_INTERVAL)

    def release(self):
        """Release the lock, if currently held. Safe to call even if
        the lock isn't held (does nothing in that case)."""
        if self._file is None:
            return

        fcntl.flock(self._file, fcntl.LOCK_UN)
        self._file.close()
        self._file = None
        logger.debug("Display lock released.")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

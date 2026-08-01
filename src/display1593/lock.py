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

logger = logging.getLogger(__name__)

DEFAULT_LOCK_PATH = "/tmp/display1593.lock"


class DisplayLock:
    """A blocking, OS-level exclusive lock.

    Only one DisplayLock (in any process, on this machine) can be
    holding the lock at the same path at a time. Calling acquire()
    will wait (block) for as long as necessary until it's the only
    holder, then return. This is used to stop two scripts from talking
    to the display's microcontrollers over serial at the same time,
    which corrupts the communication between them.

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

    def acquire(self):
        """Block until the lock is free, then acquire it.

        Opens (creating if necessary) the lock file, then calls
        fcntl.flock() to request an exclusive lock ("LOCK_EX") on it.
        Because the non-blocking flag ("LOCK_NB") is not passed, this
        call pauses here for as long as another process already holds
        the lock, and returns as soon as it's free.

        Calling this again while already held (by this same
        DisplayLock instance) does nothing, so it's safe to call more
        than once.
        """
        if self._file is not None:
            return

        logger.debug("Waiting to acquire display lock (%s)...", self.path)
        self._file = open(self.path, "w")
        fcntl.flock(self._file, fcntl.LOCK_EX)
        logger.debug("Display lock acquired.")

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

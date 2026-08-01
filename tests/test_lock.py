import threading
import time

import pytest

from display1593.lock import DisplayLock, DisplayLockTimeout


def test_lock_can_be_acquired_and_released(tmp_path):
    lock = DisplayLock(str(tmp_path / "test.lock"))

    lock.acquire()
    lock.release()

    # Should be reusable after releasing
    lock.acquire()
    lock.release()


def test_lock_as_context_manager(tmp_path):
    with DisplayLock(str(tmp_path / "test.lock")):
        pass


def test_second_holder_blocks_until_first_releases(tmp_path):
    path = str(tmp_path / "test.lock")
    lock1 = DisplayLock(path)
    lock2 = DisplayLock(path)

    lock1.acquire()
    acquired_at = []

    def try_acquire(timeout):
        lock2.acquire(timeout=timeout)
        acquired_at.append(time.monotonic())
        lock2.release()

    t = threading.Thread(target=try_acquire, kwargs={"timeout": 2})
    t.start()

    # lock2 should still be waiting - lock1 hasn't released yet
    time.sleep(0.2)
    assert t.is_alive()
    assert acquired_at == []

    release_time = time.monotonic()
    lock1.release()

    # Now lock2 should be able to proceed
    t.join(timeout=2)
    assert not t.is_alive()
    assert acquired_at[0] >= release_time


def test_second_holder_raises_after_timeout(tmp_path):
    path = str(tmp_path / "test.lock")
    lock1 = DisplayLock(path)
    lock2 = DisplayLock(path)

    lock1.acquire()
    try:
        with pytest.raises(DisplayLockTimeout):
            lock2.acquire(timeout=0.3)
    finally:
        lock1.release()

from collections import deque

import numpy as np
import pytest

from display1593.playback import load_frames, play_frames


def make_frames(n):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, 1593, 3), dtype="uint8")


def test_load_frames_npz(tmp_path):
    frames = make_frames(3)
    path = tmp_path / "frames.npz"
    np.savez_compressed(path, led_frames=frames, fps=30.0)
    assert np.array_equal(load_frames(path), frames)


def test_load_frames_csv_dir(tmp_path):
    frames = make_frames(3)
    # Filename order, not creation order
    for i in (2, 0, 1):
        np.savetxt(
            tmp_path / f"frame_{i:04d}.csv", frames[i], fmt="%d", delimiter=","
        )
    loaded = load_frames(tmp_path)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, frames)


def test_load_frames_rejects_wrong_shape(tmp_path):
    path = tmp_path / "frames.npz"
    np.savez_compressed(path, led_frames=np.zeros((3, 100, 3), "uint8"))
    with pytest.raises(ValueError):
        load_frames(path)


class FakeDisplay:
    """Records frames and show times; no hardware."""

    pipelined = True

    def __init__(self):
        self.show_slack = deque()
        self.shown = []
        self.times = []
        self._staged = None

    def clear_all(self):
        self._staged = None

    def set_all_leds(self, rgb):
        self._staged = rgb

    def show(self, t=None):
        self.shown.append(self._staged)
        self.times.append(t)
        self.show_slack.append(0.01)


def test_play_frames_loops_at_fps():
    frames = make_frames(3)
    dis = FakeDisplay()
    summary = play_frames(dis, frames, fps=50, max_frames=7)
    assert [
        np.array_equal(s, frames[i % 3]) for i, s in enumerate(dis.shown)
    ] == [True] * 7
    assert np.allclose(np.diff(dis.times), 1 / 50)
    assert summary.startswith("7 frames in")
    assert "Spare time per frame over the last 6 frames" in summary

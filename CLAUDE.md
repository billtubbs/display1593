# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python code that drives a physical LED display: 1593 WS2811 RGB LEDs, arranged
in an irregular (non-grid) pattern behind a ~4ft x 4ft translucent screen,
controlled by two Teensy 3.1 microcontrollers connected to a Raspberry Pi over
USB serial. Because the layout is irregular, the driver and every visualisation
here treat the display as a flat list of 1593 LED addresses with known (x, y)
centres and a precomputed nearest-neighbour graph, not as a 2D image buffer.

The repo is normally developed/tested on a desktop and deployed to a Raspberry
Pi (see README.md's systemd section) where the actual hardware is attached;
most scripts run fine without hardware except for the parts that call
`Display1593.connect()`.

## Commands

```bash
pip install -e .                 # install the package (editable) + deps
pip install -e '.[test]'         # + pytest
pip install -e '.[viewer]'       # + matplotlib, for fluidsim's offline viewer

pytest                           # run the whole suite (testpaths = tests/)
pytest tests/test_lock.py        # run one file
pytest tests/test_lock.py::test_second_holder_raises_after_timeout  # one test
```

Formatting/linting is via ruff (`.vscode/settings.json` runs it as the
format-on-save formatter, line length 79); there's no separate lint config
file to invoke manually beyond `ruff format` / `ruff check`.

Entry-point scripts that talk to the real hardware are run directly, e.g.
`python show_digclock.py`, `python schelling.py --n-neighbours 6`,
`python show_image.py images/monalisa.png 8`, `python fluidsim/play_fluidsim.py`.
See README.md for running `show_digclock.py` as a systemd service on the Pi.

## Project todos

Project-level tasks (not tied to one line of code) are tracked in
`TODO.md`, not in an issue tracker. File-local items stay as inline
`# TODO:` comments in the source, as they already are throughout the
codebase (e.g. `schelling.py`, `display1593.py`).

## External dependency: `serial_comm`

The low-level serial framing protocol lives in a **separate sibling repo**,
`/Users/billtubbs/ser-comm-py`, providing the `serial_comm` package
(installed here in editable mode - see its `.pth`/finder in `.venv`). It is
not vendored or a git submodule; changes to the wire protocol are made there.
Key pieces `display1593.py` relies on:

- `connect_to_arduino(ser)` - reads the board's unchecksummed hello message
  and returns `(status, name)`.
- `send_data_to_arduino` / `receive_data_from_arduino` - frame data between
  `START_MARKER`/`END_MARKER` bytes with byte-stuffing around `SPECIAL_BYTE`
  (`encode_data`/`decode_data`, numba-jitted).

## Architecture

### Core package: `src/display1593/`

- **`display1593.py`** - the `Display1593` driver class. Owns two serial
  connections (one Teensy per board, 798 + 795 LEDs), a small binary command
  protocol (`LC` clear, `L1`/`LN` set one/many LEDs, `CA`/`CN` set all/many to
  one colour, `SN` show-now), and dispatch of a flat LED-id array to the
  correct board + local LED id (`self.led_idx` cumulative boundaries; the
  numba-jitted `_board_leds`/`_board_leds_with_rgb`/`make_idx_array` do this
  split). `set_leds()`/`set_all_leds()` only *stage* values on the
  microcontrollers - nothing is actually shown until `show()`. Exposes
  the display geometry as `dis.leds` (`centres_x`/`centres_y`,
  `nearest_neighbours`, `nearest_neighbour_distances`) so callers (schelling,
  fluidsim) can do spatial reasoning without re-deriving it. Usable as a
  context manager (`with Display1593() as dis:`), which calls `connect()`/
  `disconnect()`. `show(t=None)` sends the show command now, or (with
  `t`, a `time.monotonic()` value) waits until `t` first, logging a
  warning if `t` has already passed. `connect()` drains stale bytes from
  each port's input buffer before reading the board's hello message.
  - **Synchronous vs. pipelined mode.** By default every command is sent
    and its checksummed response checked before returning, and `show()`
    displays the values just set. `Display1593(pipelined=True)` (or
    `start_pipeline()` after connecting) switches to pipelined mode, where
    **`show()` displays the frame set before the *previous* `show()`
    call** - the first frame appears at the second call, and `flush(t)`
    shows the last one (`disconnect()` flushes too). In this mode:
    `set_*()`/`clear_all()` stage commands on the Pi (`_pending`) until
    the next `show(t)`, which queues a `ShowMarker` for `t` on each
    board's `BoardSerialWorker` thread, then releases the staged frame
    behind it. Each worker sends commands in order (up to `max_inflight`
    unacknowledged); at a marker it waits until every board's worker has
    reached it (i.e. sent the frame), then until `t`, then sends `SN` -
    so both halves update together, and frame *k+1*'s data can't
    overwrite frame *k* on the boards before it's shown. Late frames
    (board not ready by `t`) and late `show(t)` calls are logged;
    `show()` blocks only when `max_frames_ahead` markers are pending;
    worker errors (e.g. `response_timeout`) are raised at the next
    `show()`/`flush()` - there's no silent fallback to synchronous mode.
    In both modes `dis.show_slack` records the spare time before each
    `show(t)`'s `t` (negative = late): pipelined, from both boards having
    received the frame; synchronous, from the `show(t)` call.
    **No entry-point script uses pipelined mode yet** - only
    `diagnostics/serial_worker_timing.py` does. README.md documents the
    user-facing behaviour, with a timeline.
- **`lock.py`** - `DisplayLock`, a `flock()`-based, wait-then-timeout,
  cross-process exclusive lock (`/tmp/display1593.lock` by default) so two
  scripts can't drive the serial connections at once; `Display1593.connect()`
  acquires it and `disconnect()` releases it.
- **`data/ledArray_data_1593.py`** - the physical layout: per-strip LED
  counts, `centres_x`/`centres_y`, and a `compute_nearest_neighbours()`
  generator (periodic/wrap-around KDTree) used to (re)produce
  `data/nearest_neighbours_1593.csv` and `data/nearest_neighbour_distances_1593.csv`.
  **`display1593.py` loads the neighbour data from those CSVs, not from this
  module's own arrays**, which have known indexing errors near the array
  edges (see the comment in `display1593.py`).
- **`image_conversion.py`** - `prepare_image()` (centre-crop/resize to
  256x256) and `convert_image()` (average each LED's precomputed source-pixel
  mask, `data/mask1593.pickle`, into one RGB triple per LED). Deliberately
  kept import-light (no `serial`/`numba`) so it can be used without the
  hardware deps - `display1593/__init__.py` lazily imports `Display1593`
  itself for the same reason.
- **`digclock1.py`** - `SevenSegmentClockFace`: pure rendering logic mapping
  a digit position (0-3) + value (0-9), or the once-per-second flash element,
  to `(led_indices, brightness_values)`, backed by `data/digdata.pickle`
  (a per-LED brightness map, tuned per physical LED, keyed by
  position -> segment -> LED). A clock **face** is a pluggable rendering
  strategy: any object providing `n_digits`, `digit_leds()`,
  `clear_indices()`, `flash_leds()` can substitute for it.
- **`logging_utils.py`** - `configure_root_logging(log_path)`, called once
  by entry-point scripts, attaches one size-capped `RotatingFileHandler` to
  the *root* logger so records from `display1593`, `display1593.lock`,
  `serial_comm`, and the script itself all land in one file.

### Entry-point scripts (repo root)

- **`show_digclock.py`** - the clock display driver: owns hardware
  connection, timing (`SecondTicker`, one tick per wall-clock second),
  day/night brightness (`BCYCLE`, keyed by hour), and compositing digit/flash
  brightness into a persistent `smem` buffer, diffed against `smem_prev` to
  push only changed LEDs (`push_changes`). Rendering itself is delegated to a
  clock face object (see `digclock1.py`). Staging (`set_leds`) happens ahead
  of a tick; the actual `show()` fires immediately after the tick so a
  minute rollover's digit change and flash toggle land in one hardware
  refresh.
- **`show_image.py`** - crops/resizes and displays a single image file.
- **`schelling.py`** - a Schelling segregation simulation running live on the
  display; each agent occupies one LED cell, and neighbour lookups use the
  display's precomputed (periodic) `nearest_neighbours` table rather than
  building a fresh KDTree over agent positions. Supports both the classic
  unweighted happiness rule and an inverse-distance-weighted one
  (`is_happy_weighted`, the default for agents here).

### `diagnostics/`

Hardware/protocol/timing scripts for debugging, not normal use. Run from
the repo root (log files are written to the current directory).

- **`check_led_neighbours.py`** - interactive terminal tool to visually
  verify the nearest-neighbour data against the physical display (lights one
  LED white, its neighbours red, step through with keypresses).
- **`async_display_smoke_test.py`** - minimal "does the Pi talk to both
  boards" check: lights a few LEDs across both boards, then clears them.
- **`compare_modes.py`** - plays the same full-display animation in
  synchronous and pipelined mode and compares max frame rate (with
  simulated per-frame work, `--compute-ms`) and, at each `--fps`, show
  time error vs. schedule, board skew, time blocked in the driver, and
  late-frame warnings. `--mock` checks the script without hardware.
- **`serial_worker_timing.py`** - per-phase timing of the pipelined-mode
  serial workers (command generation, queueing, send, response wait,
  `show()`); `--mock` runs it without hardware.
- **`comm_led_test.py`, `led_command_tests.py`,
  `frame_display_speed_test.py`** - older low-level protocol/timing tests
  (some predate the `src/display1593` package layout and import
  `serial_comm`/`display1593` more directly).

### `fireplace/`

`play_fire_frames.py` plays back precomputed per-LED RGB frames in a loop,
from `fireplace/data/fire_frames.npz` if present (a 197.7s, 5932-frame
loop generated by the separate `led-fire-place` repo's
`generate_loop_frames.py`, ~10 MB - **not committed**, copied to the Pi
with rsync), otherwise from the committed one-CSV-per-frame 30.5s clip in
`fireplace/data/`. Played at `FPS` = 24 (`--fps`;
deliberately slower than the original video's 30 fps - it looks better),
scheduling each frame with `show(t)` - in pipelined mode by default
(`--sync` for synchronous). Late frames are logged to
`fireplace/play_fire_frames.log`, with a summary printed on Ctrl+C,
including the mean and shortest spare time per frame from
`dis.show_slack`.
`test_fire_frames.py` is an older experiment that cycles a few fire images
from `images/` (currently missing from the repo, so it won't run as-is).
Kept here rather than in `diagnostics/` so `fireplace/` stays
self-contained in case it's split out of this repo.

### `fluidsim/`

A 2D buoyancy-driven ("stable fluids"-style) flow simulation over the same
1593-point layout, built with CasADi for fast repeated evaluation - separate
in spirit from the display driver (pure computation; only borrows the LED
layout geometry).

- **`fluidsim.py`** - `Geometry` builds, once, a pruned neighbour graph from
  the precomputed nearest-neighbour tables (x wraps as a period-2000 torus,
  y does not - it's a vertical container, open top/bottom) and the discrete
  operators derived from it (`Gx`, `Gy` gradient, `L` graph Laplacian), plus
  point-selection helpers for choosing which points a caller commandeers as
  boundary conditions: `points_within_radius` (a circular patch, no longer
  used by default) and `add_ghost_boundary` (off-screen points just beyond
  the real top/bottom edges, used with `one_to_one=True` - see below and
  TODO.md for the many-to-many variants that were tried and found worse).
  `NavierStokesSim` compiles one CasADi `Function` that advects/diffuses/
  buoys velocity+temperature with RK4, pressure-projects via Jacobi sweeps
  against `L`, and applies boundary conditions (free-slip top/bottom,
  fixed-temperature "commandeered" points for the heater/sink). **Note the
  y-axis convention**: `centres_y` follows image/screen convention (low y =
  physical top), so gravity/buoyancy signs are inverted relative to a normal
  math y-axis - see the module docstring before touching sign conventions
  here. `_to_casadi_sparse`'s scipy->CasADi conversion had a real,
  previously-undetected bug (wrong `invert_mapping` value, silently
  corrupting Gx/Gy/L for some sparsity patterns) - fixed; see its comment
  before changing that function.
- **`play_fluidsim.py`** - runs the sim live and pushes each frame straight
  to the LED display (temperature -> colour via a capped "plasma"-colormap
  ramp, gamma-corrected for the LEDs' non-linear response). Self-heals from
  numerical blow-up (resets to the cold initial state on NaN/Inf) since it's
  meant to run unattended. The heater/sink are off-screen ghost points just
  beyond the real top/bottom edges (`Geometry.add_ghost_boundary`, sized by
  `--ghost-offset`, defaulting to `--cutoff`) rather than any real,
  displayed LED. Current defaults (`cutoff=100`, many-to-many connectivity)
  measured **0 divergence resets over 1600s** - the best result found, at
  the cost of visibly less complex flow (one broad plume instead of several
  narrower ones) than the shorter-lived `cutoff=80` configurations - a
  wider neighbourhood radius means every point's `Gx`/`Gy`/`L` operators
  average over a physically larger area, which smooths out structure finer
  than that radius independent of the actual `nu`/`kappa` values (and
  plausibly *why* it's more stable too - the finest structure is generally
  what an explicit scheme like this struggles with). See TODO.md for the
  full comparison across boundary layouts/connectivity - **nothing tested
  is actually provably immune** to the underlying instability (every
  configuration eventually shows the same slow-overshoot-then-NaN pattern
  given enough simulated time, or hasn't been run long enough to know); the
  self-heal-and-reset is what actually makes any of this viable unattended,
  not the boundary choice. Most physics/appearance knobs (`--nu`, `--kappa`,
  `--buoyancy`, `--brightness-divisor`, `--gamma`, `--fps`) are CLI flags -
  see their `--help` text for tuned-by-eye defaults and stability notes.
  `nu`, `buoyancy`, and `kappa` were each found (via `view_fluidsim.py`
  experiments) to be fairly close to a joint numerical stability limit -
  don't assume raising one has free headroom without testing offline first.
- **`view_fluidsim.py`** - runs the same scenario on the desktop (no
  hardware), periodically saving PNG frames (via the same
  `temperature_to_rgb` mapping) and raw `.npz` state, plus diagnostic
  min/max/overshoot stats - the way to check simulation behaviour without
  the physical display.
- **`make_video.py`** - stitches `view_fluidsim.py`'s PNG frames into an MP4
  timelapse (requires `ffmpeg`); sorts by the numeric frame index parsed from
  filenames rather than by filename text.

## Working with LED data

- LED ids are a single flat index (0 to `n_leds - 1`); `Display1593.led_idx`
  gives the cumulative per-board boundaries used to split a flat id into
  (board, local id).
- Always prefer `dis.leds.nearest_neighbours` /
  `dis.leds.nearest_neighbour_distances` (the corrected CSV-backed data) over
  anything computed fresh from `ledArray_data_1593.py`'s own arrays for
  edge LEDs.
- `set_leds`/`set_all_leds`/etc. only stage LED state on the microcontrollers;
  nothing appears on the physical display until `show()` is called.

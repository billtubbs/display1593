"""
Run the fluidsim Navier-Stokes simulation live and display it on the LED
array in real time (see fireplace/play_fire_frames.py for the precedent
this follows - but that script plays back precomputed frames, while this
one computes each frame on the fly, so the achievable display rate is
capped by how fast NavierStokesSim.step() runs on this machine).

Temperature is mapped to colour with a simple cold (dark blue) -> hot
(pale yellow) ramp; there's no image data involved, unlike show_image.py.
"""

import argparse
import time
from collections import deque

import numpy as np

from display1593 import Display1593
from fluidsim import Geometry, NavierStokesSim

MAX_LOGGED_TIMES = 1000

# (temperature fraction, R, G, B) control points for the cold->hot ramp.
_COLOUR_STOPS = np.array(
    [
        [0.0, 0, 0, 40],
        [0.5, 255, 90, 0],
        [1.0, 255, 240, 180],
    ]
)


def temperature_to_rgb(T, T_cold, T_hot):
    """Map per-LED temperature to an (n_leds, 3) uint8 RGB array."""
    span = T_hot - T_cold
    u = np.clip((T - T_cold) / span, 0.0, 1.0) if span else np.zeros_like(T)
    stops, colours = _COLOUR_STOPS[:, 0], _COLOUR_STOPS[:, 1:]
    rgb = np.stack(
        [np.interp(u, stops, colours[:, c]) for c in range(3)], axis=1
    )
    return rgb.astype("uint8")


def benchmark_step(sim, n_warmup=5, n_timed=20):
    """Time NavierStokesSim.step() on this machine; returns seconds/step."""
    n = sim.geometry.n
    u = np.zeros(n)
    v = np.zeros(n)
    T = np.zeros(n)
    Tb = np.zeros(n)
    for _ in range(n_warmup):
        u, v, T = sim.step(u, v, T, Tb)
    t0 = time.perf_counter()
    for _ in range(n_timed):
        u, v, T = sim.step(u, v, T, Tb)
    return (time.perf_counter() - t0) / n_timed


def run(dis, sim, boundary_idx, T_cold, T_hot, hot_start_time, time_step):
    n = sim.geometry.n
    mask = np.zeros(n)
    mask[boundary_idx] = 1.0

    u = np.zeros(n)
    v = np.zeros(n)
    T = np.full(n, T_cold)

    dis.clear_all()

    print("Starting...")
    start_time = time.monotonic()
    scheduled_times = deque(maxlen=MAX_LOGGED_TIMES)
    actual_times = deque(maxlen=MAX_LOGGED_TIMES)
    wait_times = deque(maxlen=MAX_LOGGED_TIMES)

    step_count = 0
    next_time = time.monotonic()
    try:
        while True:
            t = step_count * sim.dt
            boundary_temp = T_hot if t >= hot_start_time else T_cold
            T_boundary = mask * boundary_temp
            u, v, T = sim.step(u, v, T, T_boundary)
            step_count += 1

            dis.set_all_leds(temperature_to_rgb(T, T_cold, T_hot))

            next_time += time_step
            scheduled_times.append(next_time)

            # Synchronize display to clock
            wait_time = max(0, next_time - time.monotonic())
            wait_times.append(wait_time)
            time.sleep(wait_time)
            dis.show_now()

            actual_times.append(time.monotonic())

    except KeyboardInterrupt:
        print("Stopped.")

    scheduled_times = np.array(scheduled_times) - start_time
    actual_times = np.array(actual_times) - start_time

    for sch, act, wait in zip(scheduled_times, actual_times, wait_times):
        print(f"{sch:6.3f} {act:6.3f} {wait * 1000:6.2f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=80.0)
    parser.add_argument("--nu", type=float, default=100.0, help="viscosity")
    parser.add_argument(
        "--kappa", type=float, default=20.0, help="thermal diffusivity"
    )
    parser.add_argument("--buoyancy", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.02, help="physics timestep")
    parser.add_argument(
        "--n-jacobi",
        type=int,
        default=40,
        help="pressure-projection Jacobi iterations (lower = faster, less "
        "accurate incompressibility)",
    )
    parser.add_argument("--heater-x", type=float, default=1000.0)
    parser.add_argument("--heater-y", type=float, default=1000.0)
    parser.add_argument("--heater-radius", type=float, default=200.0)
    parser.add_argument("--t-cold", type=float, default=0.0)
    parser.add_argument("--t-hot", type=float, default=1.0)
    parser.add_argument("--hot-start-time", type=float, default=1.0)
    parser.add_argument(
        "--time-step",
        type=float,
        default=None,
        help="seconds between LED updates (default: benchmark this machine "
        "and pick something with headroom above the measured step cost)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    geo = Geometry(cutoff=args.cutoff)
    boundary_idx = geo.points_within_radius(
        (args.heater_x, args.heater_y), args.heater_radius
    )
    print(f"heater patch: {boundary_idx.size} points")

    sim = NavierStokesSim(
        geo,
        boundary_idx,
        nu=args.nu,
        kappa=args.kappa,
        buoyancy_coeff=args.buoyancy,
        dt=args.dt,
        n_jacobi=args.n_jacobi,
    )

    step_seconds = benchmark_step(sim)
    print(
        f"measured step() time on this machine: {step_seconds * 1000:.2f} ms "
        f"({1 / step_seconds:.1f} steps/sec)"
    )

    time_step = args.time_step
    if time_step is None:
        # Leave headroom above the measured compute cost for LED
        # communication (set_all_leds/show_now), which isn't included in
        # the benchmark above.
        time_step = max(0.1, step_seconds * 2)
        print(f"--time-step not given, using {time_step:.3f}s")

    with Display1593() as dis:
        run(
            dis,
            sim,
            boundary_idx,
            args.t_cold,
            args.t_hot,
            args.hot_start_time,
            time_step,
        )


if __name__ == "__main__":
    main()

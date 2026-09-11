"""
Run the fluidsim Navier-Stokes simulation live and display it on the LED
array in real time (see fireplace/play_fire_frames.py for the precedent
this follows - but that script plays back precomputed frames, while this
one computes each frame on the fly, so the achievable display rate is
capped by how fast NavierStokesSim.step() runs on this machine).

Temperature is mapped to colour with a ramp sampled from matplotlib's
"plasma" colormap (dark blue -> purple -> red -> orange -> yellow), capped
short of plasma's own brightest tip so "hot" isn't blinding. There's no
image data involved, unlike show_image.py.

digclock.py's LEDs are always pure red (only the red channel is ever
set), so a raw value there and a raw channel value here aren't the same
amount of light: three channels lit at once is far more total output
than one. `brightness_divisor` (applied the same way as digclock's
per-hour `bness`, just fixed rather than time-varying) scales all three
channels down uniformly to compensate - tune it by eye on the actual
display, this is not something that can be derived exactly.

The WS28xx LEDs' actual light output is a non-linear (concave) function
of the programmed 0-255 value - e.g. 1->2 is the single biggest jump in
real output, with steadily diminishing returns as the value climbs toward
255. `temperature_to_rgb` treats the colour ramp below as the *intended*
(perceptually linear) brightness and gamma-corrects it (raising the
normalised 0-1 value to `gamma`, ~2.2 being the usual default for these
strips) before scaling to 0-255, so the ramp's steps look even in
practice instead of front-loaded into the low end.
"""

import argparse
import time

import numpy as np

from display1593 import Display1593
from fluidsim import DEFAULT_CUTOFF, Geometry, NavierStokesSim

# (temperature fraction, R, G, B) control points for the cold->hot ramp,
# sampled from matplotlib's "plasma" colormap at u = 0.85 * fraction (i.e.
# capped at plasma's u=0.85, short of its brightest/most saturated tip) -
# except the cold end, deliberately darkened to near-black rather than
# plasma's own (still fairly saturated) dark blue.
_COLOUR_STOPS = np.array(
    [
        [0.0, 0, 0, 12],
        [0.2, 94, 1, 166],
        [0.4, 158, 25, 157],
        [0.6, 205, 74, 118],
        [0.8, 239, 126, 80],
        [1.0, 254, 186, 44],
    ]
)


def _gamma_correct(rgb_0_255, gamma):
    """
    Compensate for the LEDs' non-linear response: treats `rgb_0_255` as
    the intended (perceptually linear) brightness and returns the
    programmed value that should actually produce it.
    """
    normalised = np.clip(rgb_0_255, 0, 255) / 255.0
    return (normalised**gamma) * 255.0


def temperature_to_rgb(T, T_cold, T_hot, brightness_divisor=1, gamma=2.2):
    """Map per-LED temperature to an (n_leds, 3) uint8 RGB array."""
    span = T_hot - T_cold
    u = np.clip((T - T_cold) / span, 0.0, 1.0) if span else np.zeros_like(T)
    stops, colours = _COLOUR_STOPS[:, 0], _COLOUR_STOPS[:, 1:]
    rgb = np.stack(
        [np.interp(u, stops, colours[:, c]) for c in range(3)], axis=1
    )
    rgb = rgb / brightness_divisor
    rgb = _gamma_correct(rgb, gamma)
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


def run(
    dis,
    sim,
    heater_idx,
    sink_idx,
    n_real,
    T_cold,
    T_hot,
    hot_start_time,
    time_step,
    brightness_divisor=1,
    gamma=2.2,
    report_interval=60.0,
):
    n = sim.geometry.n
    heater_mask = np.zeros(n)
    heater_mask[heater_idx] = 1.0
    sink_mask = np.zeros(n)
    sink_mask[sink_idx] = 1.0

    u = np.zeros(n)
    v = np.zeros(n)
    T = np.full(n, T_cold)

    dis.clear_all()

    print("Starting...")
    step_count = 0
    compute_times = []
    max_dT_history = []
    max_dvel_history = []
    report_start = time.monotonic()
    next_time = time.monotonic()

    try:
        while True:
            t = step_count * sim.dt
            heater_temp = T_hot if t >= hot_start_time else T_cold
            T_boundary = heater_mask * heater_temp + sink_mask * T_cold

            compute_start = time.perf_counter()
            u_next, v_next, T_next = sim.step(u, v, T, T_boundary)

            if (
                np.isfinite(u_next).all()
                and np.isfinite(v_next).all()
                and np.isfinite(T_next).all()
            ):
                # Per-step change is an early warning sign of instability:
                # a blow-up typically shows this growing sharply for
                # several steps before anything actually turns to NaN.
                max_dT_history.append(np.abs(T_next - T).max())
                max_dvel_history.append(
                    np.hypot(u_next - u, v_next - v).max()
                )
                u, v, T = u_next, v_next, T_next
                step_count += 1
            else:
                # This is a long-running/unattended display, so a numerical
                # blow-up (see fluidsim.py's docstring re: stability) needs
                # to self-heal rather than leave the display stuck on
                # garbage forever - NaN, once it appears, poisons every
                # later step. Discard the bad state and restart the
                # cold -> heated-at-hot_start_time cycle from scratch.
                print(
                    f"WARNING: simulation diverged at t={t:.2f}s (NaN/Inf) - "
                    "resetting to the cold initial state"
                )
                u = np.zeros(n)
                v = np.zeros(n)
                T = np.full(n, T_cold)
                step_count = 0

            # Ghost boundary points (indices >= n_real) are never part of
            # the real, displayed LEDs - slice them off before mapping to
            # colour.
            rgb = temperature_to_rgb(
                T[:n_real], T_cold, T_hot, brightness_divisor, gamma
            )
            compute_time = time.perf_counter() - compute_start
            compute_times.append(compute_time)

            dis.set_all_leds(rgb)

            # Synchronize display to a fixed-rate clock: each frame is due
            # at next_time regardless of how long the previous one took, so
            # the schedule doesn't drift even if individual frames run late.
            next_time += time_step
            wait_time = next_time - time.monotonic()
            if wait_time < 0:
                print(
                    f"WARNING: frame took {compute_time * 1000:.1f} ms to "
                    f"compute, {-wait_time * 1000:.1f} ms over the "
                    f"{time_step * 1000:.0f} ms budget for {1 / time_step:.1f} fps"
                )
                wait_time = 0
            time.sleep(wait_time)
            dis.show_now()

            now = time.monotonic()
            if now - report_start >= report_interval:
                arr = np.array(compute_times)
                print(
                    f"compute time over last {now - report_start:.1f}s "
                    f"({arr.size} frames): min={arr.min() * 1000:.2f} ms "
                    f"max={arr.max() * 1000:.2f} ms avg={arr.mean() * 1000:.2f} ms"
                )
                if max_dT_history:
                    dT_arr = np.array(max_dT_history)
                    dvel_arr = np.array(max_dvel_history)
                    print(
                        f"max per-step change over last {now - report_start:.1f}s: "
                        f"|dT| min={dT_arr.min():.4f} max={dT_arr.max():.4f} "
                        f"avg={dT_arr.mean():.4f}; "
                        f"|dvel| min={dvel_arr.min():.4f} max={dvel_arr.max():.4f} "
                        f"avg={dvel_arr.mean():.4f}"
                    )
                compute_times = []
                max_dT_history = []
                max_dvel_history = []
                report_start = now

    except KeyboardInterrupt:
        print("Stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    parser.add_argument(
        "--nu",
        type=float,
        default=300.0,
        help="viscosity - every boundary layout tried eventually diverges "
        "given enough simulated time (a slow overshoot buildup, then a "
        "sudden NaN blow-up - handled by the self-heal reset below, not "
        "eliminated by any settings found so far). The off-screen ghost "
        "boundary used here lasted 1164.6s before its first divergence "
        "(longest measured of several layouts - see TODO.md), vs. 568-"
        "682s for full-width bands or a many-to-many ghost wiring, and "
        "nu values from 150-260 diverged within 15-25s even with the "
        "small-patch layout this replaced. buoyancy=2.0 and kappa=60 "
        "both diverged within 150s at nu=300 - treat nu, buoyancy and "
        "kappa as fairly close to a joint stability limit, not "
        "independently adjustable with headroom to spare",
    )
    parser.add_argument(
        "--kappa", type=float, default=20.0, help="thermal diffusivity"
    )
    parser.add_argument("--buoyancy", type=float, default=1.0)
    parser.add_argument(
        "--brightness-divisor",
        type=int,
        default=2,
        help="fixed divisor applied to all three colour channels (like "
        "digclock's per-hour bness, but constant) - digclock's LEDs are "
        "always pure red, so an RGB heat-map lights three channels at "
        "once for the 'same' peak value and reads far brighter; tune by "
        "eye on the actual display",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="gamma-correction exponent compensating for the LEDs' "
        "non-linear response (1.0 = no correction; higher values darken "
        "low/mid brightness more while leaving near-255 mostly unchanged)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.02, help="physics timestep"
    )
    parser.add_argument(
        "--n-jacobi",
        type=int,
        default=40,
        help="pressure-projection Jacobi iterations (lower = faster, less "
        "accurate incompressibility)",
    )
    parser.add_argument(
        "--ghost-offset",
        type=float,
        default=None,
        help="distance beyond the top/bottom edge for the off-screen "
        "ghost boundary (see Geometry.add_ghost_boundary) - defaults to "
        "--cutoff, the only value validated (1164.6s to first "
        "divergence, longest of several boundary layouts tried; see "
        "TODO.md). All 1593 real LEDs are free-evolving fluid; no real "
        "LED is forced to a fixed colour, unlike the small circular "
        "patches or full-width bands tried earlier",
    )
    parser.add_argument("--t-cold", type=float, default=0.0)
    parser.add_argument("--t-hot", type=float, default=1.0)
    parser.add_argument("--hot-start-time", type=float, default=1.0)
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="fixed LED update rate - the loop paces itself to this clock "
        "regardless of compute time, warning if a frame runs over budget",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=60.0,
        help="seconds between compute-time min/max/avg reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    geo = Geometry(cutoff=args.cutoff)
    n_real = geo.n
    ghost_offset = args.ghost_offset if args.ghost_offset is not None else geo.cutoff
    heater_idx, sink_idx = geo.add_ghost_boundary(offset=ghost_offset)
    boundary_idx = np.union1d(heater_idx, sink_idx)
    print(
        f"ghost boundary: {heater_idx.size} hot + {sink_idx.size} cold "
        f"off-screen points ({n_real} real LEDs all free-evolving)"
    )

    sim = NavierStokesSim(
        geo,
        boundary_idx,
        nu=args.nu,
        kappa=args.kappa,
        buoyancy_coeff=args.buoyancy,
        dt=args.dt,
        n_jacobi=args.n_jacobi,
    )

    time_step = 1.0 / args.fps
    step_seconds = benchmark_step(sim)
    print(
        f"measured step() time on this machine: {step_seconds * 1000:.2f} ms "
        f"({1 / step_seconds:.1f} steps/sec); target is {args.fps:.1f} fps "
        f"({time_step * 1000:.0f} ms/frame budget)"
    )
    if step_seconds >= time_step:
        print(
            "WARNING: compute alone already exceeds the frame budget, before "
            "even accounting for LED communication - expect over-budget "
            "warnings once running, or lower --fps / --n-jacobi"
        )

    with Display1593() as dis:
        run(
            dis,
            sim,
            heater_idx,
            sink_idx,
            n_real,
            args.t_cold,
            args.t_hot,
            args.hot_start_time,
            time_step,
            args.brightness_divisor,
            args.gamma,
            args.report_interval,
        )


if __name__ == "__main__":
    main()

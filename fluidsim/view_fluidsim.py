"""
Run the fluidsim scenario on this desktop machine (not the LED hardware)
and periodically save snapshots for offline diagnosis.

Runs the "cold container, heated patch turns on at t=1s, cold sink held
fixed the whole time" scenario step-by-step (like play_fluidsim.py's
live loop, but with no display I/O and no real-time pacing, so it runs
as fast as this machine's step() allows) and every `--save-interval`
simulated seconds:

- prints a diagnostic line: overall/heater/sink min-max temperature and
  a count of "overshoot" points (interior points, excluding the heater
  and sink patches themselves, sitting outside [T_cold, T_hot]) - this
  is the direct way to check claims like "the cold sink isn't cooling"
  or "there are unexpectedly hot cells" without waiting on the actual
  display overnight
- saves a PNG frame coloured with the *same* temperature_to_rgb mapping
  play_fluidsim.py sends to the LEDs, so it can be compared directly
  against a photo of the physical display
- saves the raw (t, u, v, T) state as a .npz for quantitative follow-up
- appends one row per point to a single `sim_results.csv` (t, point, cx,
  cy, T - physical simulation state, not LED colour) - the long format
  makes it easy to filter/pivot a specific point's (or the sink's/
  heater's) trace over the whole run in pandas without touching the
  per-snapshot .npz files

Runs step-by-step rather than via NavierStokesSim.simulate() (which
returns the full step-by-step history in memory) so multi-hour simulated
durations don't require gigabytes of RAM - only the current state and
the periodic snapshots are kept.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection

from fluidsim import Geometry, NavierStokesSim
from play_fluidsim import temperature_to_rgb

_HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = _HERE / "data"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cutoff", type=float, default=80.0)
    parser.add_argument(
        "--nu",
        type=float,
        default=300.0,
        help="viscosity - validated stable over 80s+ at buoyancy=1.0 with "
        "the heater at the physical bottom of the display; see "
        "play_fluidsim.py's --nu help for why this is higher than you "
        "might expect",
    )
    parser.add_argument(
        "--kappa", type=float, default=20.0, help="thermal diffusivity"
    )
    parser.add_argument("--buoyancy", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--n-jacobi", type=int, default=40)
    parser.add_argument(
        "--seconds",
        type=float,
        default=3600.0,
        help="sim duration - default is 1 simulated hour, to reproduce "
        "the kind of long-run drift only seen on an overnight display run",
    )
    parser.add_argument("--heater-x", type=float, default=1000.0)
    parser.add_argument(
        "--heater-y",
        type=float,
        default=1500.0,
        help="~25%% up from the bottom of the physical display (centres_y "
        "follows image/screen convention - high y is physically low)",
    )
    parser.add_argument("--heater-radius", type=float, default=150.0)
    parser.add_argument(
        "--sink-x",
        type=float,
        default=0.0,
        help="0.0 sits on the periodic x-wrap seam (left/right edge)",
    )
    parser.add_argument(
        "--sink-y",
        type=float,
        default=500.0,
        help="~25%% down from the top of the physical display (centres_y "
        "follows image/screen convention - low y is physically high)",
    )
    parser.add_argument("--sink-radius", type=float, default=150.0)
    parser.add_argument("--t-cold", type=float, default=0.0)
    parser.add_argument("--t-hot", type=float, default=1.0)
    parser.add_argument("--hot-start-time", type=float, default=1.0)
    parser.add_argument(
        "--save-interval",
        type=float,
        default=60.0,
        help="simulated seconds between saved snapshots (PNG + .npz + a "
        "printed diagnostic line) - default is every simulated minute",
    )
    parser.add_argument(
        "--brightness-divisor",
        type=int,
        default=1,
        help="passed through to temperature_to_rgb - play_fluidsim.py "
        "uses 2 to tame the real LEDs' output, but that's a correction "
        "for physical hardware brightness, not for a PNG viewed on a "
        "monitor, so it defaults to 1 (no dimming) here",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="passed through to temperature_to_rgb - play_fluidsim.py "
        "uses ~2.2 to pre-compensate for the real LEDs' non-linear "
        "(concave) response, which would just make a screen preview "
        "look artificially dark (a monitor doesn't share that "
        "non-linearity), so it defaults to 1.0 (no correction) here",
    )
    parser.add_argument(
        "--led-diameter",
        type=float,
        default=40.0,
        help="LED marker diameter, in the layout's own coordinate units",
    )
    parser.add_argument(
        "--quiver", action="store_true", help="overlay a velocity quiver plot"
    )
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    geo = Geometry(cutoff=args.cutoff)
    heater_idx = geo.points_within_radius(
        (args.heater_x, args.heater_y), args.heater_radius
    )
    sink_idx = geo.points_within_radius(
        (args.sink_x, args.sink_y), args.sink_radius
    )
    boundary_idx = np.union1d(heater_idx, sink_idx)
    print(f"heater patch: {heater_idx.size} points")
    print(f"cold sink: {sink_idx.size} points")
    print(f"wall (floor/ceiling) points: {geo.wall_idx.size}")

    sim = NavierStokesSim(
        geo,
        boundary_idx,
        nu=args.nu,
        kappa=args.kappa,
        buoyancy_coeff=args.buoyancy,
        dt=args.dt,
        n_jacobi=args.n_jacobi,
        T_min=args.t_cold,
        T_max=args.t_hot,
    )

    n = geo.n
    heater_mask = np.zeros(n)
    heater_mask[heater_idx] = 1.0
    sink_mask = np.zeros(n)
    sink_mask[sink_idx] = 1.0
    # "Interior" for overshoot diagnostics: everywhere except the two
    # patches whose temperature is exogenously forced every step - those
    # are supposed to sit exactly at T_hot/T_cold, so including them would
    # just measure the forcing itself, not the free field's behaviour.
    interior_mask = np.ones(n, dtype=bool)
    interior_mask[heater_idx] = False
    interior_mask[sink_idx] = False

    n_total_steps = int(round(args.seconds / args.dt))
    save_every = max(1, int(round(args.save_interval / args.dt)))
    n_snapshots = n_total_steps // save_every
    n_digits = len(str(n_snapshots))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=args.dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    margin = args.led_diameter
    ax.set_xlim(geo.cx.min() - margin, geo.cx.max() + margin)
    # centres_y follows the image/screen convention (low y = top, high y =
    # bottom - see fluidsim.py's module docstring), the opposite of
    # matplotlib's default math convention, so the axis must be inverted
    # for a frame to look right-side-up next to the physical display.
    ax.set_ylim(geo.cy.max() + margin, geo.cy.min() - margin)
    ax.set_aspect("equal")
    ax.axis("off")

    offsets = np.column_stack([geo.cx, geo.cy])
    leds = EllipseCollection(
        widths=args.led_diameter,
        heights=args.led_diameter,
        angles=0,
        units="xy",
        offsets=offsets,
        offset_transform=ax.transData,
    )
    ax.add_collection(leds)
    title = ax.set_title("t = 0.00 s", color="white")

    quiver = None
    if args.quiver:
        quiver = ax.quiver(
            geo.cx,
            geo.cy,
            np.zeros(n),
            np.zeros(n),
            color="cyan",
            scale=200,
            width=0.002,
        )

    u = np.zeros(n)
    v = np.zeros(n)
    T = np.full(n, args.t_cold)

    print(f"simulating {n_total_steps} steps ({args.seconds:.1f}s)...")
    t0 = time.perf_counter()
    point_idx = np.arange(n)
    n_resets = 0

    csv_path = out_dir / "sim_results.csv"
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["t", "point", "cx", "cy", "T"])

        def save_snapshot(step, t, u, v, T):
            rgb = temperature_to_rgb(
                T,
                args.t_cold,
                args.t_hot,
                args.brightness_divisor,
                args.gamma,
            )
            leds.set_facecolor(rgb / 255.0)
            title.set_text(f"t = {t:.1f} s")
            if quiver is not None:
                quiver.set_UVC(u, v)
            idx = step // save_every
            stem = f"frame_{idx:0{n_digits}d}"
            fig.savefig(
                out_dir / f"{stem}.png", facecolor=fig.get_facecolor()
            )
            np.savez(out_dir / f"{stem}.npz", t=t, u=u, v=v, T=T)
            csv_writer.writerows(
                zip(np.full(n, t), point_idx, geo.cx, geo.cy, T)
            )
            csv_file.flush()

            overshoot = interior_mask & (
                (T > args.t_hot) | (T < args.t_cold)
            )
            elapsed = time.perf_counter() - t0
            pct = 100.0 * step / n_total_steps
            # ETA from the rate observed so far (step 0 has no rate yet).
            eta = (elapsed / step * (n_total_steps - step)) if step else 0.0
            print(
                f"[{pct:5.1f}%  elapsed={elapsed / 60:5.1f}min  "
                f"eta={eta / 60:5.1f}min]  "
                f"t={t:7.1f}s  "
                f"T overall=[{T.min():+.3f}, {T.max():+.3f}]  "
                f"sink=[{T[sink_idx].min():+.3f}, {T[sink_idx].max():+.3f}]  "
                f"heater=[{T[heater_idx].min():+.3f}, {T[heater_idx].max():+.3f}]  "
                f"overshoot points={np.count_nonzero(overshoot)}"
            )

        save_snapshot(0, 0.0, u, v, T)

        try:
            for step in range(1, n_total_steps + 1):
                t = step * args.dt
                heater_temp = (
                    args.t_hot if t >= args.hot_start_time else args.t_cold
                )
                T_boundary = (
                    heater_mask * heater_temp + sink_mask * args.t_cold
                )
                u_next, v_next, T_next = sim.step(u, v, T, T_boundary)

                if (
                    np.isfinite(u_next).all()
                    and np.isfinite(v_next).all()
                    and np.isfinite(T_next).all()
                ):
                    u, v, T = u_next, v_next, T_next
                else:
                    print(
                        f"WARNING: simulation diverged at t={t:.2f}s "
                        "(NaN/Inf) - resetting to the cold initial state"
                    )
                    u = np.zeros(n)
                    v = np.zeros(n)
                    T = np.full(n, args.t_cold)
                    n_resets += 1

                if step % save_every == 0:
                    save_snapshot(step, t, u, v, T)
        except KeyboardInterrupt:
            print("Stopped early.")

    plt.close(fig)
    elapsed = time.perf_counter() - t0
    print(
        f"wrote snapshots and {csv_path.name} to {out_dir}/ "
        f"({elapsed:.1f}s wallclock for {args.seconds:.1f}s simulated, "
        f"{n_resets} divergence reset(s))"
    )


if __name__ == "__main__":
    main()

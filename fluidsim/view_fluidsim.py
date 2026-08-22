"""
Render the fluidsim scenario as a sequence of PNG frames.

Runs the "cold container, heated patch turns on at t=1s" scenario and
saves one PNG per (strided) simulation step to an output directory, with
each LED drawn as a circle at its true diameter (in the same coordinate
units as the layout positions) coloured by temperature. This just
produces the frame sequence - turning it into a video/GIF (e.g. with
ffmpeg) is a separate step, left for later.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection

from fluidsim import Geometry, NavierStokesSim

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
        "--seconds", type=float, default=60.0, help="sim duration"
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
        "--stride", type=int, default=5, help="save every Nth simulated step"
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
    )

    n_steps = int(round(args.seconds / args.dt))
    print(f"simulating {n_steps} steps ({args.seconds:.1f}s)...")
    times, U, V, Th = sim.simulate(
        n_steps,
        heater_idx=heater_idx,
        T_cold=args.t_cold,
        T_hot=args.t_hot,
        hot_start_time=args.hot_start_time,
    )
    if np.isnan(Th).any():
        first_nan = np.argmax(np.isnan(Th).any(axis=1))
        print(
            f"WARNING: simulation went unstable (NaN) at t={times[first_nan]:.2f}s "
            "- try increasing --nu/--kappa or decreasing --dt/--buoyancy"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    finite = np.isfinite(Th)
    vmin = (
        min(args.t_cold, np.nanmin(Th[finite]))
        if finite.any()
        else args.t_cold
    )
    vmax = (
        max(args.t_hot, np.nanmax(Th[finite])) if finite.any() else args.t_hot
    )

    fig, ax = plt.subplots(figsize=(7, 7), dpi=args.dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    margin = args.led_diameter
    ax.set_xlim(geo.cx.min() - margin, geo.cx.max() + margin)
    ax.set_ylim(geo.cy.min() - margin, geo.cy.max() + margin)
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
        cmap="inferno",
    )
    leds.set_clim(vmin, vmax)
    ax.add_collection(leds)
    title = ax.set_title("t = 0.00 s", color="white")

    quiver = None
    if args.quiver:
        quiver = ax.quiver(
            geo.cx, geo.cy, U[0], V[0], color="cyan", scale=200, width=0.002
        )

    frame_steps = np.arange(0, times.size, args.stride)
    n_digits = len(str(frame_steps.size - 1))
    for i, step in enumerate(frame_steps):
        leds.set_array(Th[step])
        title.set_text(f"t = {times[step]:.2f} s")
        if quiver is not None:
            quiver.set_UVC(U[step], V[step])
        fig.savefig(
            out_dir / f"frame_{i:0{n_digits}d}.png",
            facecolor=fig.get_facecolor(),
        )

    plt.close(fig)
    print(f"wrote {frame_steps.size} frames to {out_dir}/")


if __name__ == "__main__":
    main()

"""Visualize the full neighbour graph (fluid points + grid ghost points),
including every real-real and real-ghost connection as a line."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from fluidsim import DEFAULT_CUTOFF, Geometry, PERIOD  # noqa: E402

layout = sys.argv[1] if len(sys.argv) > 1 else "grid"

geo = Geometry(cutoff=DEFAULT_CUTOFF)
n_real = geo.n
bottom_ghost_idx, top_ghost_idx = geo.add_ghost_boundary(layout=layout)
bottom_set = set(bottom_ghost_idx.tolist())

wall_bottom = geo.wall_idx[geo.cy[geo.wall_idx] > geo.cy.mean()]
wall_top = geo.wall_idx[geo.cy[geo.wall_idx] <= geo.cy.mean()]
interior = np.setdiff1d(np.arange(n_real), geo.wall_idx)

# Real-real edges: drawn once per unique pair, skipping periodic x-wrap
# pairs (a straight line to the naive cx position would cut clean across
# the whole domain width, which is misleading - those points are only
# "close" because x wraps, not because they're actually near each other
# on this flat drawing).
real_segments = []
for i in range(n_real):
    js, dx, dy = geo._neighbours[i]
    for j in js:
        if j <= i or j >= n_real:
            continue
        if abs(geo.cx[j] - geo.cx[i]) > PERIOD / 2:
            continue
        real_segments.append([(geo.cx[i], geo.cy[i]), (geo.cx[j], geo.cy[j])])

# Real-ghost edges (grid ghosts never wrap in x relative to a nearby real
# edge point, since both sit well inside one period of each other).
hot_segments, cold_segments = [], []
for i in geo.wall_idx:
    js, dx, dy = geo._neighbours[i]
    for j in js[js >= n_real]:
        seg = [(geo.cx[i], geo.cy[i]), (geo.cx[j], geo.cy[j])]
        (hot_segments if j in bottom_set else cold_segments).append(seg)

print(f"layout={layout}: {len(real_segments)} real-real edges, "
      f"{len(hot_segments)} real-hotghost, {len(cold_segments)} real-coldghost")

fig, ax = plt.subplots(figsize=(7, 7))

ax.add_collection(LineCollection(real_segments, colors="#bbbbbb", linewidths=0.3, zorder=0))
ax.add_collection(LineCollection(hot_segments, colors="#f4a582", linewidths=0.5, zorder=1))
ax.add_collection(LineCollection(cold_segments, colors="#92c5de", linewidths=0.5, zorder=1))

ax.scatter(geo.cx[interior], geo.cy[interior], s=8, c="#444444", zorder=2, label="fluid points (interior)")
ax.scatter(geo.cx[wall_bottom], geo.cy[wall_bottom], s=14, c="#1f77b4", zorder=3, label="fluid points w/ ghost below")
ax.scatter(geo.cx[wall_top], geo.cy[wall_top], s=14, c="#2ca02c", zorder=3, label="fluid points w/ ghost above")
ax.scatter(geo.cx[bottom_ghost_idx], geo.cy[bottom_ghost_idx], s=35, c="#d62728", marker="x", zorder=4, label=f"hot ghosts ({bottom_ghost_idx.size})")
ax.scatter(geo.cx[top_ghost_idx], geo.cy[top_ghost_idx], s=35, c="#9467bd", marker="x", zorder=4, label=f"cold ghosts ({top_ghost_idx.size})")

ax.set_ylim(geo.cy.max() + 100, geo.cy.min() - 100)
ax.set_xlim(geo.cx.min() - 20, geo.cx.max() + 20)
ax.set_aspect("equal")
ax.set_title(f"Full neighbour graph: fluid points + {layout} ghost boundary")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=7)
plt.tight_layout()
out_path = _HERE / f"full_mesh_{layout}.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)

# A zoomed-in inset near the bottom edge, where the ghost connections are
# easiest to actually see amid the full mesh.
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.add_collection(LineCollection(real_segments, colors="#bbbbbb", linewidths=0.6, zorder=0))
ax2.add_collection(LineCollection(hot_segments, colors="#f4a582", linewidths=1.0, zorder=1))
ax2.scatter(geo.cx[interior], geo.cy[interior], s=25, c="#444444", zorder=2)
ax2.scatter(geo.cx[wall_bottom], geo.cy[wall_bottom], s=40, c="#1f77b4", zorder=3, label="fluid pts w/ ghost")
ax2.scatter(geo.cx[bottom_ghost_idx], geo.cy[bottom_ghost_idx], s=90, c="#d62728", marker="x", zorder=4, label="hot ghosts")
ax2.set_xlim(0, 500)
ax2.set_ylim(geo.cy.max() + 60, geo.cy.max() - 200)
ax2.set_aspect("equal")
ax2.set_title(f"Zoomed: bottom edge detail ({layout} layout)")
ax2.legend(loc="upper right", fontsize=8)
plt.tight_layout()
out_path2 = _HERE / f"full_mesh_{layout}_zoom.png"
fig2.savefig(out_path2, dpi=150)
print("saved", out_path2)

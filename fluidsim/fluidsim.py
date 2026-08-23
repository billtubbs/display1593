"""
2D buoyancy-driven flow simulation on the irregular 1593-LED grid.

Standalone from the `display1593` package (it only borrows the LED layout
geometry from there) - this is pure computation, no LED rendering, no
hardware connection.

Geometry
--------
The layout tiles left-right as a 2000-wide torus in x (confirmed against
nearest_neighbour_distances_1593.csv: wrapped point-to-point distances
computed with that period reproduce the recorded distances exactly). y
does NOT wrap: the domain represents a vertical container of fluid - open/
free at top and bottom, wrapping only side to side, like a cross-section
of an ocean. `Geometry` builds, once, from the precomputed
18-nearest-neighbour tables:

Note on the y-axis direction: `centres_y` follows the image/screen
convention, not the math/plot one - confirmed empirically by running a
top-red/bottom-blue test image through `Display1593.convert_image()` and
checking which `centres_y` range came back which colour (see the
conversation this was found in, or just rerun that check if you doubt
it): low `centres_y` is the TOP of the physical display, high `centres_y`
is the BOTTOM. Gravity therefore acts in +y here, and buoyancy (hot fluid
rising) acts in -y - the reverse of what you'd assume from a normal
math/plot y-axis.

- a pruned neighbour graph keeping only pairs closer than `cutoff` (default
  80 units - chosen so most points keep a full ~6-neighbour first ring
  while discarding the more distant second ring), with any pair that only
  counted as "close" via a top-to-bottom wrap removed
- `Gx`, `Gy`: sparse discrete gradient operators, from a per-point weighted
  least-squares fit of (df/dx, df/dy) to neighbour value differences
- `L`: a sparse discrete Laplacian, the standard inverse-distance-squared
  weighted graph Laplacian (`L[i,i] = -sum_j w_ij`, `L[i,j] = w_ij`) - this
  form is exactly diagonally dominant by construction, which is what makes
  the Jacobi pressure-projection in `NavierStokesSim` stable
- `wall_idx`: points that lost a neighbour to the top/bottom-wrap pruning,
  i.e. the top and bottom edge of the container (a floor/ceiling)

Simulation
----------
`NavierStokesSim` uses those operators to build one CasADi function that
advances the whole grid's (u, v, T) state by one explicit timestep:
advect + diffuse + buoyancy-force (hot fluid rises, buoyancy acts in -y -
see the y-axis note above) the provisional velocity, project it onto an
(approximately) divergence-free field via a fixed number of Jacobi sweeps
against `L`,
diffuse/advect temperature, then enforce boundary conditions: zero
vertical velocity at `wall_idx` (free-slip floor/ceiling), and zero
velocity plus an exogenous, time-varying temperature at a fixed,
caller-chosen set of "commandeered" points (e.g. a heating patch).

This is a simplified/artistic model (a "stable fluids"-style scheme
adapted to a point cloud), not a research-grade Navier-Stokes solver:
good enough to look right, not meant to be quantitatively accurate.
"""

from pathlib import Path

import casadi as ca
import numpy as np
import scipy.sparse as sp

from display1593.data import ledArray_data_1593 as _layout
from display1593.data.ledArray_data_1593 import centres_x, centres_y

_DATA_DIR = Path(_layout.__file__).resolve().parent

# The layout tiles left-right as a torus of this width (verified: using
# this period to unwrap x-offsets reproduces the recorded distances in
# nearest_neighbour_distances_1593.csv exactly). y does not wrap.
PERIOD = 2000.0

DEFAULT_CUTOFF = 80.0


def _wrap(delta, period=PERIOD):
    """Map a coordinate difference to its shortest signed periodic equivalent."""
    return (delta + period / 2) % period - period / 2


class Geometry:
    """
    Static geometry for the 1593-point grid: positions, a pruned neighbour
    graph (periodic in x, open in y), and the discrete gradient/Laplacian
    operators built from it. Everything here depends only on the LED
    layout, not on any simulated field, so it's computed once and reused.
    """

    def __init__(self, cutoff=DEFAULT_CUTOFF):
        self.cx = np.asarray(centres_x, dtype=float)
        self.cy = np.asarray(centres_y, dtype=float)
        self.n = self.cx.size

        nbrs_full = np.loadtxt(
            _DATA_DIR / "nearest_neighbours_1593.csv",
            delimiter=",",
            dtype=np.int64,
        )
        dist_full = np.loadtxt(
            _DATA_DIR / "nearest_neighbour_distances_1593.csv", delimiter=","
        )

        neighbours = self._pruned_neighbours(nbrs_full, dist_full, cutoff)
        self.Gx, self.Gy, self.L = self._build_operators(neighbours)
        self.L_diag = np.asarray(self.L.diagonal())

    def _pruned_neighbours(self, nbrs_full, dist_full, cutoff):
        """
        For each point, (neighbour_indices, dx, dy) within `cutoff`.

        x wraps periodically (the layout tiles left-right), but y does
        not: any neighbour pair whose raw y-difference required a
        periodic correction (i.e. would only be "close" by wrapping
        top-to-bottom) is dropped. A point that loses a neighbour this
        way sits on the top or bottom edge of the container; those are
        recorded in `self.wall_idx` so the simulation can pin their
        vertical velocity to zero (a floor/ceiling).
        """
        keep = dist_full < cutoff
        neighbours = []
        wall_mask = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            js = nbrs_full[i, keep[i]]
            dy_raw = self.cy[js] - self.cy[i]
            dy = _wrap(dy_raw)
            no_y_wrap = np.abs(dy_raw - dy) < PERIOD / 4
            if not no_y_wrap.all():
                wall_mask[i] = True
            js = js[no_y_wrap]
            dy = dy[no_y_wrap]
            dx = _wrap(self.cx[js] - self.cx[i])
            if js.size < 2:
                raise ValueError(
                    f"point {i} has fewer than 2 neighbours after pruning "
                    f"(cutoff={cutoff}); the gradient fit needs at least 2 "
                    "non-collinear neighbours per point"
                )
            neighbours.append((js, dx, dy))

        # nearest_neighbours_1593.csv is a plain per-point k-NN table, not
        # an enforced-mutual one, so cutoff-pruning could in principle
        # leave one-way pairs (i lists j but not vice versa) - Gx/Gy/L are
        # built assuming a symmetric graph, so fail loudly rather than
        # silently build inconsistent operators.
        neighbour_sets = [set(js.tolist()) for js, _, _ in neighbours]
        one_way = [
            (i, j)
            for i, (js, _, _) in enumerate(neighbours)
            for j in js.tolist()
            if i not in neighbour_sets[j]
        ]
        if one_way:
            raise ValueError(
                f"{len(one_way)} one-way neighbour pair(s) after pruning "
                f"(cutoff={cutoff}), e.g. {one_way[:5]} - the "
                "gradient/Laplacian operators assume a symmetric graph"
            )

        self.wall_idx = np.nonzero(wall_mask)[0]
        return neighbours

    def _build_operators(self, neighbours):
        n = self.n
        rows, cols, gx_vals, gy_vals, lap_vals = [], [], [], [], []
        diag_gx = np.zeros(n)
        diag_gy = np.zeros(n)
        diag_lap = np.zeros(n)

        for i, (js, dx, dy) in enumerate(neighbours):
            r2 = dx**2 + dy**2
            w = 1.0 / r2

            # Weighted least-squares fit of (df/dx, df/dy) at point i from
            # f_j - f_i ~ fx*dx + fy*dy, weighted by inverse distance^2 so
            # closer neighbours dominate the fit.
            A = np.stack([dx, dy], axis=1)
            AtW = A.T * w
            coeff = np.linalg.solve(AtW @ A, AtW)  # shape (2, k)

            rows.extend([i] * js.size)
            cols.extend(js.tolist())
            gx_vals.extend(coeff[0].tolist())
            gy_vals.extend(coeff[1].tolist())
            diag_gx[i] = -coeff[0].sum()
            diag_gy[i] = -coeff[1].sum()

            # Inverse-distance-squared weighted graph Laplacian.
            lap_vals.extend(w.tolist())
            diag_lap[i] = -w.sum()

        rows = np.asarray(rows)
        cols = np.asarray(cols)
        Gx = sp.coo_matrix((gx_vals, (rows, cols)), shape=(n, n)) + sp.diags(
            diag_gx
        )
        Gy = sp.coo_matrix((gy_vals, (rows, cols)), shape=(n, n)) + sp.diags(
            diag_gy
        )
        L = sp.coo_matrix((lap_vals, (rows, cols)), shape=(n, n)) + sp.diags(
            diag_lap
        )
        return Gx.tocsr(), Gy.tocsr(), L.tocsr()

    def points_within_radius(self, center, radius):
        """Indices of points within `radius` of `center` (x-wrap-aware)."""
        cx0, cy0 = center
        dx = _wrap(self.cx - cx0)
        dy = self.cy - cy0
        return np.nonzero(np.hypot(dx, dy) <= radius)[0]


def _to_casadi_sparse(mat):
    """scipy.sparse matrix -> constant casadi.DM with the same sparsity/values."""
    coo = mat.tocoo()
    n, m = coo.shape
    pattern, mapping = ca.Sparsity.triplet(
        n, m, coo.row.tolist(), coo.col.tolist(), True
    )
    data = np.asarray(coo.data)[np.asarray(mapping)]
    return ca.DM(pattern, data.tolist())


class NavierStokesSim:
    """
    Explicit, "stable fluids"-style time-stepper for 2D buoyancy-driven
    flow on a `Geometry`'s irregular grid, built once as a single CasADi
    Function for fast repeated evaluation.

    The domain is a vertical container: periodic side-to-side (x), open
    top and bottom (y), with gravity acting in +y (`centres_y` follows the
    image/screen convention - low y is the top of the physical display,
    high y is the bottom - see the module docstring), so buoyancy from a
    temperature above `T_ref` acts in -y - hot fluid rises. `wall_idx`
    (from `geometry`, the top/bottom edge points) get a free-slip floor/
    ceiling condition: vertical velocity pinned to zero, horizontal
    velocity free. `boundary_idx` is a separate, caller-chosen set of
    "commandeered" points held at zero velocity and an exogenous,
    time-varying temperature (e.g. a heating patch) - only the
    temperature *values* there vary per `step()` call; the set of indices
    is fixed at construction.
    """

    def __init__(
        self,
        geometry,
        boundary_idx,
        nu=5.0,
        kappa=5.0,
        buoyancy_coeff=0.5,
        T_ref=0.0,
        dt=0.05,
        n_jacobi=40,
    ):
        self.geometry = geometry
        self.boundary_idx = np.asarray(boundary_idx)
        self.dt = dt
        self.n_jacobi = n_jacobi

        n = geometry.n
        fixed_mask = np.zeros(n)
        fixed_mask[self.boundary_idx] = 1.0
        wall_mask = np.zeros(n)
        wall_mask[geometry.wall_idx] = 1.0
        free_mask = 1.0 - fixed_mask

        Gx = ca.SX(_to_casadi_sparse(geometry.Gx))
        Gy = ca.SX(_to_casadi_sparse(geometry.Gy))
        L = ca.SX(_to_casadi_sparse(geometry.L))
        L_diag = ca.DM(geometry.L_diag)

        fixed = ca.DM(fixed_mask)
        free = ca.DM(free_mask)
        wall = ca.DM(wall_mask)

        u = ca.SX.sym("u", n)
        v = ca.SX.sym("v", n)
        T = ca.SX.sym("T", n)
        Tb = ca.SX.sym("Tb", n)

        def rhs(u, v, T):
            """
            Advection + diffusion + buoyancy, i.e. everything except the
            pressure term - the "d/dt" of (u, v, T) that a plain forward
            Euler step would use.

            Forward Euler applied directly to this (central-difference
            advection + explicit time-stepping) is a classic unconditionally
            unstable combination (FTCS instability) - fine for a while, then
            blows up exponentially regardless of how small dt is. RK4 (see
            below) is what actually makes this stable.
            """
            adv_u = u * (Gx @ u) + v * (Gy @ u)
            adv_v = u * (Gx @ v) + v * (Gy @ v)
            adv_T = u * (Gx @ T) + v * (Gy @ T)
            # centres_y follows the image/screen convention (low y = top,
            # high y = bottom - see the module docstring), so "up" is -y
            # and hot fluid's buoyant acceleration is subtracted here, not
            # added.
            buoyancy = buoyancy_coeff * (T - T_ref)
            du = -adv_u + nu * (L @ u)
            dv = -adv_v + nu * (L @ v) - buoyancy
            dT = -adv_T + kappa * (L @ T)
            return du, dv, dT

        # Classic RK4 for advection/diffusion/buoyancy; the pressure
        # projection (an algebraic constraint, not part of the ODE) is
        # applied once, after the full RK4 step, not per stage.
        k1u, k1v, k1T = rhs(u, v, T)
        k2u, k2v, k2T = rhs(
            u + dt / 2 * k1u, v + dt / 2 * k1v, T + dt / 2 * k1T
        )
        k3u, k3v, k3T = rhs(
            u + dt / 2 * k2u, v + dt / 2 * k2v, T + dt / 2 * k2T
        )
        k4u, k4v, k4T = rhs(u + dt * k3u, v + dt * k3v, T + dt * k3T)

        u_star = u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        v_star = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        T_new = T + dt / 6 * (k1T + 2 * k2T + 2 * k3T + k4T)

        # Free-slip floor/ceiling, applied before the pressure solve so the
        # divergence it corrects already reflects the wall.
        v_star = (1 - wall) * v_star

        div_star = Gx @ u_star + Gy @ v_star
        pressure_rhs = div_star / dt

        # Jacobi iterations for the pressure Poisson equation L@p = rhs.
        # Pressure is pinned to zero at the commandeered patch, which acts
        # as a fixed (zero-velocity) inclusion and needs a reference point
        # to anchor the otherwise arbitrary-up-to-a-constant solution.
        p = ca.SX.zeros(n)
        for _ in range(n_jacobi):
            Lp = L @ p
            p = (pressure_rhs - (Lp - L_diag * p)) / L_diag
            p = free * p

        u_new = u_star - dt * (Gx @ p)
        v_new = (1 - wall) * (v_star - dt * (Gy @ p))

        u_new = free * u_new
        v_new = free * v_new
        T_new = fixed * Tb + free * T_new

        self._step_fn = ca.Function(
            "ns_step",
            [u, v, T, Tb],
            [u_new, v_new, T_new],
            ["u", "v", "T", "T_boundary"],
            ["u_next", "v_next", "T_next"],
        )

    def step(self, u, v, T, T_boundary):
        """Advance (u, v, T) by one timestep of size `self.dt`."""
        u_next, v_next, T_next = self._step_fn(u, v, T, T_boundary)
        return (
            np.asarray(u_next).ravel(),
            np.asarray(v_next).ravel(),
            np.asarray(T_next).ravel(),
        )

    def simulate(
        self,
        n_steps,
        heater_idx=None,
        T_cold=0.0,
        T_hot=1.0,
        hot_start_time=1.0,
        u0=None,
        v0=None,
        T0=None,
    ):
        """
        Run the scenario: all points start at `T_cold` and stationary.
        Every point in `self.boundary_idx` is held at `T_cold` throughout
        *except* those also in `heater_idx` (default: all of
        `boundary_idx`), which switch to `T_hot` once `hot_start_time`
        seconds have elapsed - so a subset outside `heater_idx` (e.g. a
        cold sink) stays fixed at `T_cold` for the whole run. Returns
        (times, U, V, Th), each of shape (n_steps + 1, n_points) except
        `times`.
        """
        n = self.geometry.n
        fixed_mask = np.zeros(n)
        fixed_mask[self.boundary_idx] = 1.0
        heater_mask = np.zeros(n)
        heater_mask[
            self.boundary_idx if heater_idx is None else heater_idx
        ] = 1.0
        sink_mask = fixed_mask - heater_mask

        u = np.zeros(n) if u0 is None else np.array(u0, dtype=float)
        v = np.zeros(n) if v0 is None else np.array(v0, dtype=float)
        T = np.full(n, T_cold) if T0 is None else np.array(T0, dtype=float)

        times = np.arange(n_steps + 1) * self.dt
        U = np.empty((n_steps + 1, n))
        V = np.empty((n_steps + 1, n))
        Th = np.empty((n_steps + 1, n))
        U[0], V[0], Th[0] = u, v, T

        for step in range(1, n_steps + 1):
            t = step * self.dt
            heater_temp = T_hot if t >= hot_start_time else T_cold
            T_boundary = heater_mask * heater_temp + sink_mask * T_cold
            u, v, T = self.step(u, v, T, T_boundary)
            U[step], V[step], Th[step] = u, v, T

        return times, U, V, Th

# TODO

Project-level tasks that don't belong to one specific line of code (see also
inline `# TODO:` comments in the source for smaller, file-local items).

- [ ] Wire up the remaining firmware commands documented in
      `led_commands.md` (in the sibling `ser-comm-py` repo) that
      `Display1593` doesn't expose yet: `LB` (set brightness), `G1` (get LED
      colour), `GB` (get photoresistor brightness), `GT` (get clock time),
      `SA` (show at scheduled clock time), `RR` (report ready to show).
      `SA`/`RR` in particular look like the intended mechanism for the
      cross-board sync noted as a TODO in `Display1593.show_now()`.
- [ ] `fluidsim`: replace `NavierStokesSim`'s explicit (RK4) advection with
      Jos Stam-style semi-Lagrangian advection (trace each point back by
      `-dt*(u,v)` and interpolate from its `nearest_neighbours`, using the
      existing precomputed neighbour tables). Unlike the current scheme,
      semi-Lagrangian advection is unconditionally stable regardless of
      `dt`, which would remove the CFL-style ceiling that currently forces
      small `dt`/high `nu` for stability. Bigger rewrite, not a quick
      tuning change - see `fluidsim/fluidsim.py`'s module docstring for
      why RK4 is needed today.
- [x] `fluidsim`: replace the small circular heater/sink patches
      (`points_within_radius`) with an off-screen "ghost boundary"
      (`Geometry.add_ghost_boundary(offset=geo.cutoff, one_to_one=True)`),
      so all 1593 real LEDs are free-evolving fluid instead of ~89 real
      LEDs pinned to a fixed hot/cold colour. Now wired into
      `play_fluidsim.py`/`view_fluidsim.py` as the default (`--ghost-
      offset` to override). **Important caveat, found the hard way
      tonight: this is not immune to the underlying numerical
      instability** - see below. It's the best of everything tried, not
      a fix for the root cause (that's the semi-Lagrangian item above).

      One ghost point per real point in `self.wall_idx` (points that
      lost a neighbour to the top/bottom y-wrap pruning in
      `_pruned_neighbours` - see that method's docstring for why a full
      vertical wrap isn't used instead, which would remove the floor/
      ceiling buoyancy needs), each fixed hot/cold via the existing
      `boundary_idx`/`fixed_mask` mechanism, never displayed.
      `play_fluidsim.py`/`view_fluidsim.py` slice `T`/`u`/`v` down to the
      first `n_real` (1593) points before mapping to LED colour or
      logging, since ghosts aren't part of the real display.

      Found and fixed a real, previously-latent bug in `_to_casadi_sparse`
      along the way: `ca.Sparsity.triplet(..., invert_mapping=True)` was
      silently producing a wrong Gx/Gy/L matrix for the ghost-augmented
      sparsity pattern (rows with a single nonzero) - confirmed by direct
      comparison against the dense scipy matrix. Fixed to `invert_mapping
      =False`. Verified the original 1593-point matrix converts
      correctly either way, so this did not affect any of tonight's
      patches/bands testing - only the in-progress ghost work.

      **Connectivity structure matters more than expected, and the
      "more correct" answer measured worse.** Every point in the real
      graph normally gets several neighbours found by a cutoff-radius
      search, not a fixed 1:1 pairing - so a proper ghost boundary
      "should" wire each real edge point to every nearby ghost, and
      each ghost to every nearby real point, the same way. Tried
      exactly that (`one_to_one=False`, the default) two ways, at
      `nu=300`/`kappa=20`/`buoyancy=1.0`/`dt=0.02`/300s runs:
        - Mirror layout (one ghost generated per edge point, then a
          cutoff-radius search for connections): 2.8 ghosts/point avg -
          diverges every ~73s.
        - Each real edge point's *actual* recorded cross-edge neighbour
          distance(s) (`self._y_wrap_pruned`, from the raw doubly-
          periodic KDTree data in `ledArray_data_1593.py`'s
          `compute_nearest_neighbours()`): 2.1 ghosts/point avg -
          diverges every ~76s.
        - Grid layout (`layout="grid"`: an evenly-spaced row of ghosts,
          independent of the real edge's irregular spacing): 1.2
          ghosts/point avg, but each ghost serves ~3.3 real points -
          held stable to 300s, but diverged once at t=682s over an 800s
          run.
      Versus **`one_to_one=True`** (each edge point wired to exactly the
      one ghost generated from it, `add_ghost_boundary`'s current
      default connectivity): diverged once at **t=1164.6s** in a 1600s
      run - the longest measured time-to-first-divergence of everything
      tried, including patches and bands. The pattern held across every
      many-to-many variant: more simultaneous ghost connections per
      point made things worse, not better, likely because a fixed
      (Dirichlet) neighbour dominates a point's local weighted-least-
      squares gradient fit more the more of them there are.
      `one_to_one` reconstructs this via the actual per-point
      construction correspondence, not "nearest ghost by distance" -
      wall points aren't perfectly evenly spaced, so a neighbour's ghost
      is occasionally geometrically closer than a point's own, which
      would silently wire up a different (untested, and worse) pairing.

      **No boundary layout tested tonight is actually immune** - patches
      (800s+, untested beyond that), full-width bands (diverges every
      570-680s), and every ghost variant above all eventually hit the
      same failure pattern: a slow overshoot buildup, then a sudden NaN
      blow-up. They only differ in how long that takes. The existing
      self-heal-and-reset in `play_fluidsim.py`/`view_fluidsim.py` (not
      any boundary choice) is what actually makes this viable
      unattended. Raising `nu` to 400 only delayed bands' divergence by
      ~20% (568s -> 683s), not removed it.

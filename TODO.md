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
- [ ] `fluidsim`: finish validating `Geometry.add_ghost_boundary()` and,
      if it holds up, wire it into `play_fluidsim.py`/`view_fluidsim.py`
      in place of the small circular heater/sink patches
      (`points_within_radius`), so all 1593 real LEDs are free-evolving
      fluid instead of ~55 real LEDs pinned to a fixed hot/cold colour.

      **Status as of tonight**: implemented in `fluidsim/fluidsim.py`
      (`Geometry.add_ghost_boundary`), reusing `self.wall_idx` (points
      that lost a neighbour to the top/bottom y-wrap pruning in
      `_pruned_neighbours` - see that method's docstring for why a full
      vertical wrap isn't used instead, which would remove the floor/
      ceiling buoyancy needs). Each real edge point gets one off-screen
      ghost, fixed hot/cold via the existing `boundary_idx`/`fixed_mask`
      mechanism, never displayed. `play_fluidsim.py`/`view_fluidsim.py`
      would need to slice `T`/`u`/`v` down to the first 1593 (real)
      points before mapping to LED colour, since ghosts aren't part of
      the real display.

      Found and fixed a real, previously-latent bug in `_to_casadi_sparse`
      along the way: `ca.Sparsity.triplet(..., invert_mapping=True)` was
      silently producing a wrong Gx/Gy/L matrix for the ghost-augmented
      sparsity pattern (rows with a single nonzero) - confirmed by direct
      comparison against the dense scipy matrix. Fixed to `invert_mapping
      =False`. Verified the original 1593-point matrix converts
      correctly either way, so this did not affect any of tonight's
      patches/bands testing - only the in-progress ghost work.

      Ghost ***offset*** matters a lot and isn't obvious - three variants
      tried, tested via a scratch script (not yet in the repo) mirroring
      `view_fluidsim.py`'s divergence-tracking loop:
        - offset = DEFAULT_CUTOFF/2 (40): diverges almost immediately.
        - offset = DEFAULT_CUTOFF (80), one ghost/point: **zero
          divergence over 300s** - the current default in
          `add_ghost_boundary`.
        - Each real edge point's *actual* recorded cross-edge neighbour
          distance(s) (`self._y_wrap_pruned`, from the raw doubly-
          periodic KDTree data in `ledArray_data_1593.py`'s
          `compute_nearest_neighbours()`) - more principled in theory,
          but most points have 2-4 such pairs, and multiple simultaneous
          ghost connections per point seems to over-weight the boundary
          condition locally; diverges every ~76s. Worse than the flat
          80-unit single-ghost version, despite being "more correct"
          geometrically.

      **Not yet done**: an 800s+ validation run at offset=80 (matching
      the level of scrutiny given to the patches below) - start there
      before trusting this for a live/overnight run.

      Related finding, independent of the ghost work: at
      `nu=300`/`kappa=20`/`buoyancy=1.0`/`dt=0.02`, **full-width heater/
      sink bands diverge roughly every 570-680 simulated seconds** (a
      slow overshoot buildup, then a sudden NaN blow-up, then the
      existing self-heal resets to cold) - confirmed via
      `python fluidsim/view_fluidsim.py` (no args) reproducing this
      live. The small circular patches (`points_within_radius`,
      `play_fluidsim.py`'s current default) were the only boundary
      layout confirmed stable over the same duration (0 divergence
      resets over 800s, tested standalone). Raising `nu` to 400 only
      delayed the bands' divergence by ~20% (568s -> 683s), not removed
      it - not a fix on its own. This is why `play_fluidsim.py` is back
      on patches for now rather than bands.

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
      (`Geometry.add_ghost_boundary`), so all 1593 real LEDs are
      free-evolving fluid instead of ~89-133 real LEDs pinned to a fixed
      hot/cold colour. Wired into `play_fluidsim.py`/`view_fluidsim.py`.
      **Important caveat, found the hard way tonight: this is not
      immune to the underlying numerical instability** - see below.
      It's the best of everything tried, not a fix for the root cause
      (that's the semi-Lagrangian item above).

      **Current deployed default, now properly validated:**
      `cutoff=DEFAULT_CUTOFF=100` (up from 80; both scripts import this
      from `fluidsim.py` instead of separately hardcoding their own
      `--cutoff` default - that duplication had silently drifted apart
      once, worth watching for again), `offset=geo.cutoff` (=100, both
      scripts' own default when `--ghost-offset` isn't passed), with
      many-to-many connectivity (`one_to_one=False`, `add_ghost_
      boundary`'s default). **0 divergence resets over the full 3600s
      (1 simulated hour)** - the best result of the whole session,
      beating patches, bands, and every other ghost variant below.
      Confirmed twice: a standalone script mirroring `view_fluidsim.
      py`'s test loop ran clean to 1600s with a self-correcting wobble
      around t=1080-1300s (overshoot rose to 18 points, speed peaked
      ~10.7, then recovered on its own - the first configuration all
      night to show that kind of recovery instead of either staying
      clean or diverging outright); a real `python fluidsim/
      view_fluidsim.py --save-interval 15` run on the actual Mac went
      the full 3600s with 0 resets, showing an equivalent wobble at
      t=1110-1320s (overshoot up to 16 points) that also self-
      corrected. Same phenomenon on both runs, slightly different
      timing/magnitude - consistent with the kind of chaotic
      sensitivity to minor floating-point differences already seen
      elsewhere tonight, not a discrepancy to worry about.

      **Trade-off, seen directly on the display, not just in
      diagnostics: visual complexity dropped noticeably** - e.g. one
      broad plume instead of the 3 narrower, hotter ones the less-
      stable `cutoff=80` configs produced. Best explanation: raising
      `cutoff` widens the neighbourhood every point's weighted-least-
      squares `Gx`/`Gy` and graph-Laplacian `L` are built from (~6.5 ->
      ~11.9 avg neighbours), which acts as extra implicit numerical
      diffusion/smoothing independent of the actual `nu`/`kappa`
      values - narrower/higher-wavenumber structure (multiple thin
      plumes) gets averaged away once it's finer than the neighbourhood
      radius. That's plausibly *also* why it's more stable: the finest,
      highest-wavenumber modes are generally the ones an explicit
      scheme like this struggles with most, so suppressing them via
      coarser resolution may be removing the specific failure mode
      responsible for every divergence tonight - a real trade of visual
      richness for stability, not a free improvement.

      An offset mismatch nearly produced a false negative here: a
      first attempt at this test used `add_ghost_boundary()`'s own
      internal default (`cutoff/2=50`) rather than what the scripts
      actually pass (`offset=geo.cutoff=100`) - diverged almost
      immediately (t=1.36s, 4207 resets in 1600s, since weight scales
      as `1/distance²` so offset=50 is a 4x stronger coupling). Always
      check a standalone test's `add_ghost_boundary(...)` call matches
      the live scripts' actual arguments, not the function's own
      defaults, before trusting a result.

      **Bug found and fixed 2026-09-10 (after this section was
      written): `layout="grid"` could silently leave a wall point with
      zero ghost connections** if it sat far enough from the true edge
      that no ghost in the fixed-offset row was within `cutoff` -
      confirmed visually (some edge points in `fluidsim/scratch/
      plot_full_mesh.py`'s diagram had no real-ghost line) and in code
      (57 of 133 wall points at cutoff=100, 26 of 89 at cutoff=80).
      **This means the "grid layout: diverged once at t=682s" result
      below was measuring a boundary with ~29% of its edge silently
      disconnected, not a real grid boundary - treat that number as
      invalid, not as evidence about the grid layout itself.** Fixed
      by falling back to each orphaned point's single nearest ghost
      (see `add_ghost_boundary`'s `else` branch) - `layout="grid"` has
      not been re-tested for stability since.

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

- [x] **Set `--nu 150` for more interesting/energetic flow** (tested
      2026-09-10 night into 2026-09-11 morning while the user slept,
      applied 2026-09-11 morning at the user's request - both
      `play_fluidsim.py` and `view_fluidsim.py` now default to
      `nu=150`). Motivation:
      at `cutoff=100`, the fluid looked visibly calmer/less complex than
      earlier, less-stable configurations (see the `cutoff=100` write-up
      above) - `cutoff=100`'s extra implicit numerical diffusion
      (wider Gx/Gy/L neighbourhoods) plausibly gives headroom to lower
      the *explicit* `nu` back down without losing the stability gained,
      since the earlier "150-260 diverges in 15-25s" finding was
      measured at `cutoff=80`, which no longer reflects the current
      setup.

      **Screening (300s each, kappa=20/buoyancy=1.0/dt=0.02/n_jacobi=40,
      offset=cutoff=100, matching the live deployed config exactly):**
      all four of nu=150/200/250/275 ran clean (0 resets), but peak
      speed shows a sharp, threshold-like jump between 200 and 250:

      | nu  | max speed @300s | overshoot @300s |
      |-----|-----------------|------------------|
      | 150 | 34.8            | 4                |
      | 200 | 8.4             | 0                |
      | 250 | 0.40            | 0                |
      | 275 | 0.39            | 0                |

      250/275 hadn't really started convecting yet by 300s (consistent
      with nu=300's own very slow ~hundreds-of-seconds ramp-up seen
      earlier tonight) - 150/200 are the candidates actually offering
      more energetic motion than the current nu=300 default within a
      reasonable timeframe. nu=150's speed (34.8) is higher than
      anything validated all night (nu=300's own peak has stayed in the
      ~10-20 range) - promising for "more interesting," but also the
      single biggest departure from anything trusted so far, so it
      needs the same longer-duration scrutiny as everything else before
      being trusted, not just a clean 300s.

      **1600s confirmatory runs:** `nu=200` diverged once at t=988.14s.
      `nu=150` ran clean (0 resets) with a self-correcting wobble around
      t=735-885s (overshoot rose to 47 points, the largest self-
      correction seen all session, then fully cleared). Counter-
      intuitively, the *lower*-viscosity `nu=150` outperformed `nu=200`:
      looking at `nu=200`'s trajectory, it never reaches a fully
      saturated convective state - it oscillates, ramping to speed ~8-13
      then partially collapsing back toward near-zero (0.187 at
      t=1020s) and re-ramping; its divergence sits right in one of
      those collapse/re-ramp transitions. `nu=150` instead switches on
      fast (full speed by t=135s) and stays in a robust, saturated
      state - plausibly why it's more, not less, stable despite less
      damping.

      **Full 3600s confirmatory run for `nu=150` (matching the exact
      scrutiny given to the current `nu=300` default): 0 divergence
      resets.** Sustained speed ~30-35 throughout (`nu=300`'s own peak
      stayed in the ~10-20 range) - genuinely more energetic, not just
      briefly. After the single wobble at t=735-885s clears (by
      t=1035s), overshoot stays at exactly 0 for the entire remaining
      ~2700s to t=3600s - even cleaner long-term than `nu=300`'s own
      3600s run (which had its own comparable wobble around t=1110-
      1320s). This is now validated to the same standard as every other
      number in this file - recommended for use.

      Test script used throughout: a standalone script mirroring
      `view_fluidsim.py`'s loop, parametrized by `nu` and duration,
      always constructing `add_ghost_boundary(offset=geo.cutoff)` to
      exactly match what `play_fluidsim.py`/`view_fluidsim.py` actually
      pass (not `add_ghost_boundary()`'s own internal default) - not
      yet committed to the repo, lives only in this session's
      scratchpad. Worth moving into `fluidsim/scratch/` (matching
      `plot_full_mesh.py`'s precedent) if more parameter sweeps like
      this are anticipated, so the "match the live scripts' actual
      arguments, not the library's internal defaults" lesson isn't
      re-learned by hand each time.

- [ ] Fix unbounded frame-pacing drift in `play_fluidsim.py`'s `run()`.
      `compute_time` (checked against the `--fps` budget, e.g. 200ms at
      fps=5) only measures `sim.step()` + `temperature_to_rgb()` - it
      excludes `dis.set_all_leds()` and `dis.show_now()`, the actual
      serial writes to the Teensy boards, which are not free (all 1593
      LEDs' RGB values over a 57600-baud link, every frame). Observed
      live on the Pi Zero 2W: `compute_time` alone was a healthy 153-
      156ms (under the 200ms budget), yet the printed "over budget"
      figure was already 6500+ ms and climbing by a few ms every frame.
      Cause: `next_time += time_step` runs on a fixed schedule from
      whenever the loop started and never resyncs - if the *true*
      per-frame cost (compute + serial I/O, not just compute) exceeds
      the budget by even a small, persistent margin, the deficit
      compounds forever with no recovery mechanism. Not urgent (doesn't
      crash anything, the display just drifts further behind real-time
      the longer it runs), but worth fixing: either resync `next_time`
      to `time.monotonic() + time_step` once the deficit exceeds some
      threshold (accept a genuinely lower effective fps rather than an
      ever-growing one), and/or include the serial write time in what's
      checked against the budget, and/or throttle the warning to once
      per `report_interval` instead of every single frame once behind.
      Also worth reconsidering whether `cutoff=100`'s higher compute
      cost (153-156ms vs. the earlier `cutoff=80` measurement of 78-
      117ms) leaves enough real-world margin at `fps=5` on the Pi Zero
      2W once serial I/O is properly accounted for - the true budget
      margin is smaller than the `compute_time`-only number suggests.

      **Practical mitigation applied 2026-09-11: `--fps` default lowered
      to 4.5** (from 5.0). From the observed drift rate in the printed
      warnings (growing ~1.5ms/frame against the 200ms/5fps budget), the
      true sustainable frame period on the Pi at this configuration is
      ~201.5ms (~4.96fps) - just barely over 5fps's budget, which is
      exactly why it drifted slowly rather than blowing up outright.
      4.5fps (222ms budget) gives ~20ms of margin over that measurement.
      This doesn't fix the resync bug above (the code will still drift
      unboundedly if the true cost ever exceeds 222ms again - e.g. if
      `cutoff`/`n_jacobi` change) - it just moves the target far enough
      below current measured cost that the bug's effect is negligible in
      practice. The actual resync fix is still worth doing.

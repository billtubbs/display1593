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

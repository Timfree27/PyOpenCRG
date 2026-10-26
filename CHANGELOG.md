# Changelog

## 0.2.1

Fixes

- Modifiers of a part of a file: `f.read(u=..., apply_mods=True)` and
  `crg.read(path, u=..., apply_mods=True)` compute the modifiers once on the whole file, so every
  part is placed exactly where it is in the modified whole road (two parts that overlap give the
  same x, y, z and normal). Before, `crg.read(path, u=..., apply_mods=True)` moved the part
  relative to its own start. `road.apply_mods()` on a part now gives a warning.
- Parts of a curved road give exactly the values of the whole road. Before, a short part of a gentle
  curve could be taken as straight, and the first and last cells of a part were slightly off the
  reference line of the whole road (below 1 mm).

New features

- `f.evaluator(window=40.0, margin=10.0, apply_mods=False)`: evaluator of an open file for
  simulations on very large files. Only a window of the road is in memory; it is read again when a
  point comes close to one of its ends. The results are the ones of the whole road (borders,
  smoothing zones and closed tracks included), and a point is never evaluated outside the window.
- `benchmarks/bench_evaluation.py`: time of the evaluation methods for 1, 640 and 10 000 points.

Improvements

- Faster evaluation of the points inside the grid (most points of a simulation):
  `ev.xy_to_z(x, y, with_normal=True)` is 1.6 times faster (640 points: 0.29 ms -> 0.18 ms),
  `uv_to_z` up to 1.8 times faster.

## 0.2.0

New features

- `crg.open(path)`: open a file once, then read parts of it with `f.read(u=..., step=...)`, like a
  Python file. Binary files stay on the disk; reading a part of a large file takes a few milliseconds.
  `crg.read()` works as before.
- Surface normal and slopes: `road.uv_to_normal()`, `road.xy_to_normal()`, `road.uv_to_gradient()`,
  and `with_normal=True` in `uv_to_z()` / `xy_to_z()` to get the height and the normal in one pass.
  They are exact derivatives of the interpolation, including slope, banking and road curvature.
- `road.evaluator()`: an evaluator prepared once for fast repeated calls (simulation loops). It starts
  each search from the last position of the points.
- `dtype="float32"` in `crg.read()` and `f.read()` to halve the memory used by the elevation grid.

Improvements

- A single `u_hint` value is enough: the hint of every point is refined automatically
  (`xy_to_uv` on 640 points spread over 60 cm: 5 times faster).
- Lower fixed cost of every evaluation call.
- `xy_to_uv` works without scipy (numpy search of the closest reference line point).

Installation

- Only numpy is required. scipy, matplotlib, PyVista and Pillow are optional:
  `pip install "pyopencrg[all]"` installs everything (recommended), `pip install pyopencrg` only the
  core. A missing package gives an error message with the command to install it.

## 0.1.0

First public release.

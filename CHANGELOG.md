# Changelog

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

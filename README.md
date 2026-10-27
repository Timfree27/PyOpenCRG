# PyOpenCRG

> **Unofficial** implementation of the OpenCRG file format. Not affiliated with, endorsed by or supported by ASAM e.V.
> OpenCRG® is a registered trademark of ASAM e.V.

![Belgian block road rendered with PyOpenCRG](https://raw.githubusercontent.com/Timfree27/PyOpenCRG/main/docs/images/belgian_block_3d.jpg)

<sub>Belgian block sample road of the ASAM OpenCRG project, drawn with `road.view3d(color="grid", cmap="jet", z_scale=2.5)` (relief exaggerated 2.5 times).</sub>

**PyOpenCRG** (`import pyopencrg`) reads, analyses, edits, shows and writes [OpenCRG](https://www.asam.net/standards/detail/opencrg/)
road surface files (`.crg`) in pure Python. You do not need to compile anything, and it runs on Windows, Linux
and macOS.

- **All OpenCRG formats**: binary `KRBI` / `KDBI` and text `LRFI` / `LDFI`, with full header support
  (comments, options, modifiers, map projection, WGS84).
- **Fast**: everything is vectorised with numpy. A file of 10 million values is read in 0.1 s
  (binary) or about 4 s (text).
- **Accurate**: data stays in double precision (float64). The algorithms follow the OpenCRG reference tools.
- **Simple**: one object, `CrgRoad`, with plain-named methods. Every editing method returns a new road.
- **Complete**: everything in the OpenCRG MATLAB tools, plus tools to clean laser scans, convert STL meshes
  and export to OBJ / STL / CSV / images / Simpack.

French guide: [docs/guide_fr.md](https://github.com/Timfree27/PyOpenCRG/blob/main/docs/guide_fr.md). Equivalence with the MATLAB tools:
[docs/matlab_equivalence.md](https://github.com/Timfree27/PyOpenCRG/blob/main/docs/matlab_equivalence.md).

## Installation

```bash
pip install "pyopencrg[all]"     # recommended: everything (scipy, matplotlib, PyVista, Pillow)
pip install pyopencrg            # light: only numpy (read, check, evaluate, edit, write)
```

The light installation is useful in a simulation or server environment (e.g. conda, where mixing pip
and conda packages can conflict). The other features need an extra package; if it is missing, the
error message gives the command to install it:

| Extra | Package | Needed for |
|---|---|---|
| `scipy` | scipy | smoothing, filters, re-sampling of curved roads, scan repair, point cloud import |
| `plot` | matplotlib | figures (`road.show()`, `road.plot...`) and image export |
| `viewer` | PyVista | interactive 3D viewer `road.view3d()` |
| `image` | Pillow | `engrave_image()` |

## In 10 lines

```python
import pyopencrg as crg

road = crg.read("my_road.crg")          # any OpenCRG file
print(road)                              # summary: length, width, grid, slope...
print(road.check())                      # consistency checks

z = road.uv_to_z(120.0, 0.5)             # height 120 m along the road, 0.5 m left of the reference line
x, y = road.uv_to_xy(120.0, 0.5)         # where this point is in the world

part = road.crop(u=(100, 200))           # keep 100 m of road (a NEW road, `road` is unchanged)
part.save("part.crg", fmt="LDFI")        # save as a text file
road.show()                              # overview figures
road.view3d()                            # interactive 3D view (PyVista)
```

Files are **checked when they are read**: a truncated, unreadable or badly defined file raises a
`CrgCheckError` explaining the problem, and minor problems are shown once as a warning. Use
`crg.read(path, check=False)` to open a damaged file anyway (to repair it), or `check="strict"` to
refuse files with warnings too.

**Large files**: only what you ask for is loaded.

```python
preview = crg.read("scan.crg", step=4)                  # one point out of 4 in both directions (16 x less memory)
detail = crg.read("scan.crg", u=(500, 520))             # 20 m at full resolution
light = crg.read("scan.crg", dtype="float32")           # half the memory for the elevation grid

with crg.open("scan.crg") as f:                         # open once, like a Python file...
    for u0 in range(0, int(f.length), 100):
        part = f.read(u=(u0, u0 + 100))                 # ...then read parts in a few milliseconds
```

A part gives exactly the same x, y, z as the whole road. With `apply_mods=True` (in `f.read()` or
`crg.read(path, u=...)`), the modifiers are computed on the whole file, so the part is placed
exactly where it is in the modified whole road.

## Tyre contact and simulation

The surface normal and the slopes are exact derivatives of the interpolation (grid, slope, banking
and curvature of the road included). For a simulation loop, an evaluator prepares everything once:

```python
nx, ny, nz = road.xy_to_normal(x, y)                     # unit normal, (0, 0, 1) on a flat road
dz_du, dz_dv = road.uv_to_gradient(u, v)                 # slopes along and across the road

ev = road.evaluator()                                    # prepared once
for step in range(n_steps):
    z, nx, ny, nz = ev.xy_to_z(x, y, with_normal=True)   # height and normal of 640 contact points: ~0.2 ms
```

The evaluator remembers the last position of each point and starts the next search from there.
Give `u_hint` (one value, e.g. the u of the wheel) to `road.xy_to_uv()` / `road.xy_to_z()` when the
road passes several times close to the same place.

**Files larger than the memory**: `f.evaluator()` has the same methods and keeps only a window of
the road in memory. The window is read again (a few milliseconds) when a point comes closer than
`margin` to one of its ends; the results are the ones of the whole road.

```python
with crg.open("track.crg") as f:
    ev = f.evaluator(window=40.0, margin=10.0, apply_mods=True)   # 40 m in memory
    for step in range(n_steps):
        z, nx, ny, nz = ev.xy_to_z(x, y, with_normal=True)
```

Speed of every method: `python benchmarks/bench_evaluation.py`.

## Gallery

Height map of the whole road, and a zoom where every 1 cm grid point is visible
(`road.plot.heightmap(cmap="jet")`; zooming in a figure re-samples the image automatically):

![Height map](https://raw.githubusercontent.com/Timfree27/PyOpenCRG/main/docs/images/belgian_block_heightmap.png)

Reference line, valid data limits, cross sections and long sections (`road.plot.show_cuts_and_limits()`):

![Cuts and limits](https://raw.githubusercontent.com/Timfree27/PyOpenCRG/main/docs/images/belgian_block_sections.png)

<sub>Pictures rendered from the sample `belgian_block.crg` of the [ASAM OpenCRG project](https://github.com/asam-ev/OpenCRG)
(data provided by DaimlerChrysler, 2005; Apache License 2.0). The sample file is not distributed with PyOpenCRG;
the pictures are made with [docs/make_images.py](https://github.com/Timfree27/PyOpenCRG/blob/main/docs/make_images.py).</sub>

## Coordinates in one picture

OpenCRG describes a road along a **reference line** (usually the road centre):

* `u` = distance along the reference line (m), `v` = lateral distance to it (m, **positive to the left**);
* `x, y, z` = position in the world.

The data is a grid `road.z` of shape `(nu, nv)`: `road.z[i, j]` is the height at `u = road.u[i]`,
`v = road.v[j]`, **relative to the reference line**. The absolute height of the road is

```
z_absolute = road.z  +  road.zr (reference line elevation)  +  road.banking * v
```

`road.uv_to_z(u, v)` always returns the absolute height.

## What you can do

| Task | Code |
|---|---|
| Read / write | `crg.read(path)`, `crg.open(path)`, `road.save(path, fmt="KRBI")`, `crg.convert(a, b, "LDFI")` |
| Information | `print(road)`, `road.check()`, `road.stats()` |
| Elevation, position | `uv_to_z`, `uv_to_xy`, `xy_to_uv`, `xy_to_z`, `u_to_phi`, `u_to_curvature` |
| Normal, slopes | `uv_to_normal`, `xy_to_normal`, `uv_to_gradient`, `xy_to_z(..., with_normal=True)` |
| Fast repeated calls | `ev = road.evaluator()`, then `ev.xy_to_z(...)`, `ev.uv_to_z(...)`... |
| Simulation on a huge file | `with crg.open(path) as f: ev = f.evaluator(window=40.0)` |
| Profiles | `road.long_section(v=0.9)`, `road.cross_section(u=50)` |
| Cut, join, flip | `crop(u=..., v=...)`, `trim(1, 1)`, `crg.concatenate([r1, r2])`, `flip()` |
| Re-sample | `resample(u_inc=0.05, v_inc=0.05)` |
| Slope / banking | `extract_slope()`, `extract_banking()`, `separate_slope_banking()`, `set_banking()` |
| Modifiers | `road.mods`, `apply_mods()`, `move_to(x, y, z, phi)`, `transform(...)`, `scale(...)` |
| Filter | `filter("gauss", 5)`, `clip_z(-0.05, 0.05)` |
| Repair scans | `remove_artifacts(0.3, 15)`, `fill_holes("linear")`, `fill_nan()`, `find_peaks()` |
| Synthetic roads | `crg.generate.synthetic_road(...)`, `from_function`, `road_from_points`, `trapezoid_bumps` |
| Import | `crg.importers.from_stl(path)`, `from_point_cloud(x, y, z)` |
| Add features | `add_profile(...)`, `add_random(...)`, `overlay_uv(other)`, `engrave_image("logo.png")` |
| Export | `to_obj`, `to_stl`, `to_csv`, `to_trm` (Simpack), `save_heightmap`, `to_html_map` |
| Geodesy | `xy_to_wgs84`, `xyz_to_llh`, `llh_to_xyz`, `crg.wgs84.distance(...)` |
| Plots | `road.show()`, `road.plot.heightmap()` (zoom shows every grid point), `road.plot.cross_sections()`, ... |
| 3D viewer | `road.view3d(color="z" / "grid" / "normal", z_scale=5)` (millions of points, GPU) |

Every function has a detailed docstring: type `help(road.crop)` in Python.

## Command line

```bash
pyopencrg info  road.crg
pyopencrg check road.crg
pyopencrg show  road.crg
pyopencrg view  road.crg                          # 3D viewer (PyVista)
pyopencrg convert road.crg road_text.crg --fmt LDFI
pyopencrg export road.crg road.obj --step 2       # .obj .stl .csv .png .html .trm
```

## Examples

See the [examples](https://github.com/Timfree27/PyOpenCRG/tree/main/examples) folder: reading and plotting, editing, synthetic roads, cleaning a laser
scan, STL to CRG conversion, an obstacle test track.

## Modifiers: a word of warning

The `$ROAD_CRG_MODS` block of a file tells a simulator how to modify the road when it loads it
(scale it, fill NaN values, move it). `crg.read()` returns the road **as stored in the file**; call
`road.apply_mods()` (or `crg.read(path, apply_mods=True)`) to get what a simulator sees. For a part
of a file, use `crg.read(path, u=..., apply_mods=True)` or `f.read(u=..., apply_mods=True)`: the
modifiers are then computed on the whole file (`part.apply_mods()` would move the part relative to
its own start). If a file
has **no** modifier block, OpenCRG applies default modifiers: NaN values are replaced by the last
valid value of the cross section and the road start is moved to x = y = z = 0 with heading 0.

## Development

```bash
pip install -e ".[test,image]"
pytest
```

## Citing

If you use pyopencrg in a report, a publication or another software, please cite it
(see [CITATION.cff](https://github.com/Timfree27/PyOpenCRG/blob/main/CITATION.cff)):

> Hardy, T. *PyOpenCRG: OpenCRG road surface files in pure Python*, version 0.2.2, 2026.

## License

Apache License 2.0, see [LICENSE](https://github.com/Timfree27/PyOpenCRG/blob/main/LICENSE) and [NOTICE](https://github.com/Timfree27/PyOpenCRG/blob/main/NOTICE). Parts of the algorithms are derived
from the ASAM OpenCRG MATLAB tools (Apache 2.0).

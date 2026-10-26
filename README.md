# pyopencrg

> **Unofficial** implementation of the OpenCRG file format. Not affiliated with, endorsed by or supported by ASAM e.V.

**pyopencrg** reads, analyses, edits, shows and writes [OpenCRG](https://www.asam.net/standards/detail/opencrg/)
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

French guide: [docs/guide_fr.md](https://github.com/Timfree27/pyopencrg/blob/main/docs/guide_fr.md). Equivalence with the MATLAB tools:
[docs/matlab_equivalence.md](https://github.com/Timfree27/pyopencrg/blob/main/docs/matlab_equivalence.md).

## Installation

```bash
pip install pyopencrg            # numpy, scipy and matplotlib are installed automatically
pip install "pyopencrg[image]"   # adds Pillow, needed only for engrave_image()
pip install "pyopencrg[viewer]"  # adds PyVista, for the fast interactive 3D viewer road.view3d()
```

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

**Files larger than the memory**: only what you ask for is loaded.

```python
preview = crg.read("scan.crg", step=4)        # one point out of 4 in both directions (16 x less memory)
detail = crg.read("scan.crg", u=(500, 520))   # 20 m at full resolution
```

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
| Read / write | `crg.read(path)`, `road.save(path, fmt="KRBI")`, `crg.convert(a, b, "LDFI")` |
| Information | `print(road)`, `road.check()`, `road.stats()` |
| Elevation, position | `uv_to_z`, `uv_to_xy`, `xy_to_uv`, `xy_to_z`, `u_to_phi`, `u_to_curvature` |
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

See the [examples](https://github.com/Timfree27/pyopencrg/tree/main/examples) folder: reading and plotting, editing, synthetic roads, cleaning a laser
scan, STL to CRG conversion, an obstacle test track.

## Modifiers: a word of warning

The `$ROAD_CRG_MODS` block of a file tells a simulator how to modify the road when it loads it
(scale it, fill NaN values, move it). `crg.read()` returns the road **as stored in the file**; call
`road.apply_mods()` (or `crg.read(path, apply_mods=True)`) to get what a simulator sees. If a file
has **no** modifier block, OpenCRG applies default modifiers: NaN values are replaced by the last
valid value of the cross section and the road start is moved to x = y = z = 0 with heading 0.

## Development

```bash
pip install -e ".[test,image]"
pytest
```

## Citing

If you use pyopencrg in a report, a publication or another software, please cite it
(see [CITATION.cff](https://github.com/Timfree27/pyopencrg/blob/main/CITATION.cff)):

> Hardy, T. *pyopencrg: OpenCRG road surface files in pure Python*, version 0.1.0, 2026.

## License

Apache License 2.0, see [LICENSE](https://github.com/Timfree27/pyopencrg/blob/main/LICENSE) and [NOTICE](https://github.com/Timfree27/pyopencrg/blob/main/NOTICE). Parts of the algorithms are derived
from the ASAM OpenCRG MATLAB tools (Apache 2.0).

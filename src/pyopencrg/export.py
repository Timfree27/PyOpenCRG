"""
Export of a road to other formats: 3D meshes, tables, images, maps, trajectories.

==========================  ===================================================
:func:`to_obj`              Wavefront OBJ mesh (Blender, game engines...)
:func:`to_stl`              binary STL mesh (CAD, 3D printing)
:func:`to_csv`              table of the grid nodes (Excel, ...)
:func:`to_trm`              Simpack trajectory of the reference line
:func:`save_heightmap`      image of the elevation grid
:func:`save_normalmap`      image of the normal map
:func:`save_combined_map`   normal map and height map stacked, in slices
:func:`to_html_map`         web page showing the road on OpenStreetMap
==========================  ===================================================
"""

from __future__ import annotations

import os

import numpy as np

from ._optional import need

__all__ = ["to_obj", "to_stl", "to_csv", "to_trm", "save_heightmap", "save_normalmap",
           "save_combined_map", "to_html_map"]


def _mesh(road, step=1, u=None):
    sub = road if u is None else road.crop(u=u)
    iu = np.arange(0, sub.nu, step)
    iv = np.arange(0, sub.nv, step)
    if iu[-1] != sub.nu - 1:
        iu = np.append(iu, sub.nu - 1)
    if iv[-1] != sub.nv - 1:
        iv = np.append(iv, sub.nv - 1)
    uu, vv = np.meshgrid(sub.u[iu], sub.v[iv], indexing="ij")
    x, y = sub.uv_to_xy(uu, vv)
    z = sub.surface_z()[np.ix_(iu, iv)]
    return x, y, z


def _triangles(z):
    nr, nc = z.shape
    idx = np.arange(nr * nc).reshape(nr, nc)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    fin = np.isfinite(z).ravel()
    ok = fin[a] & fin[b] & fin[c] & fin[d]
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    return np.concatenate((np.column_stack((a, c, d)), np.column_stack((a, d, b))))


def to_obj(road, path, step=1, u=None):
    """Export the road surface as a Wavefront OBJ mesh.

    Parameters
    ----------
    step : int
        Keep one grid point out of ``step`` (to make the file smaller).
    u : (float, float), optional
        Export only this part of the road (m).
    """
    x, y, z = _mesh(road, step, u)
    tri = _triangles(z)
    verts = np.column_stack((x.ravel(), y.ravel(), np.nan_to_num(z.ravel())))
    with open(os.fspath(path), "w") as fh:
        fh.write("# OpenCRG road exported by pyopencrg\n")
        np.savetxt(fh, verts, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, tri + 1, fmt="f %d %d %d")


def to_stl(road, path, step=1, u=None):
    """Export the road surface as a binary STL mesh (see :func:`to_obj` for the parameters)."""
    x, y, z = _mesh(road, step, u)
    tri = _triangles(z)
    p = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    v0, v1, v2 = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data = np.zeros(tri.shape[0], dtype=dt)
    data["n"] = n
    data["v"] = np.stack((v0, v1, v2), axis=1)
    with open(os.fspath(path), "wb") as fh:
        fh.write(b"OpenCRG road exported by pyopencrg".ljust(80, b" "))
        fh.write(np.uint32(tri.shape[0]).tobytes())
        fh.write(data.tobytes())


def to_csv(road, path, step=1, sep=","):
    """Export the grid nodes to a CSV table.

    Columns: ``u, v, x, y, z`` (absolute elevation) and ``z_grid``.
    """
    iu = np.arange(0, road.nu, step)
    iv = np.arange(0, road.nv, step)
    uu, vv = np.meshgrid(road.u[iu], road.v[iv], indexing="ij")
    x, y = road.uv_to_xy(uu, vv)
    zs = road.surface_z()[np.ix_(iu, iv)]
    zg = road.z[np.ix_(iu, iv)]
    table = np.column_stack([a.ravel() for a in (uu, vv, x, y, zs, zg)])
    np.savetxt(os.fspath(path), table, delimiter=sep, fmt="%.6f",
               header=sep.join(("u", "v", "x", "y", "z", "z_grid")), comments="")


_TRM_HEADER = """header.begin

    data.type   = 3      ! File format type: x, y, z, u
    data.par(1) = 1      ! Length scaling factor for x, y, z (meters)
    data.par(2) = 1      ! Length scaling factor for u (millimeters)
    data.par(4) = 0      ! Superelevation about Track centerline
    data.par(5) = 1      ! Superelevation reference baselength (meters)
    data.par(7) = 1      ! Increment for data reduction

header.end

! x              y                z              u

"""


def to_trm(road, path, step=1.0, v=0.0, extrapolation=0.0, z="zero", offset_file=True):
    """Export a trajectory along the road as a Simpack ``.trm`` file.

    Parameters
    ----------
    step : float
        Distance between the trajectory points (m).
    v : float
        Lateral position of the trajectory (m, 0 = reference line).
    extrapolation : float
        Straight length added before the start and after the end (m).
    z : str
        Height of the points: ``"zero"``, ``"reference"`` (reference line
        elevation) or ``"surface"`` (road surface).
    offset_file : bool
        Also write ``<name>_offset.txt`` with the position of the road start
        relative to the first trajectory point.
    """
    u = np.arange(road.u_start, road.u_end + 1e-9, step)
    if u[-1] < road.u_end - 1e-9:
        u = np.append(u, road.u_end)
    if extrapolation > 0:
        n = int(np.ceil(extrapolation / step))
        u = np.concatenate((road.u_start - np.arange(n, 0, -1) * step, u, road.u_end + np.arange(1, n + 1) * step))
    x, y = road.uv_to_xy(u, np.full(u.size, float(v)))
    if z == "zero":
        zz = np.zeros(u.size)
    elif z == "reference":
        zz = np.interp(u, road.u, road.zr)
    elif z == "surface":
        zz = road.uv_to_z(u, np.full(u.size, float(v)))
    else:
        raise ValueError("z must be 'zero', 'reference' or 'surface'")
    path = os.fspath(path)
    with open(path, "w") as fh:
        fh.write(_TRM_HEADER)
        for xi, yi, zi in zip(x, y, zz):
            fh.write(f"{xi:.6f}\t{yi:.6f}\t{zi:.6f}\t0\n")
    if offset_file:
        base = path[:-4] if path.lower().endswith(".trm") else path
        with open(base + "_offset.txt", "w") as fh:
            fh.write("Position of the OpenCRG road start relative to the first trajectory point:\n")
            fh.write(f"\t- x : {road.x[0] - x[0]:.6f}\n\t- y : {road.y[0] - y[0]:.6f}\n")
            fh.write(f"Road start in the trajectory frame: x = {road.x[0]:.6f}, y = {road.y[0]:.6f}\n")


def _image(a):
    """Grid (nu, nv) -> image rows = v (left side at the top), columns = u."""
    return np.ma.masked_invalid(a.T[::-1])


def save_heightmap(road, path, cmap="Greys", absolute=False):
    """Save the elevation grid as an image (u to the right, left side of the road at the top)."""
    plt = need("matplotlib.pyplot", "the image export")
    plt.imsave(os.fspath(path), _image(road.surface_z() if absolute else road.z), cmap=cmap)


def save_normalmap(road, path, cmap="Blues"):
    """Save the normal map (vertical component of the surface normal) as an image."""
    plt = need("matplotlib.pyplot", "the image export")
    plt.imsave(os.fspath(path), _image(road.normal_map()), cmap=cmap)


def save_combined_map(road, path, n_slices=1):
    """Save the normal map (top) and the height map (bottom) in one image.

    Parameters
    ----------
    n_slices : int
        Cut the image in this many slices along u: ``name0.png``, ``name1.png``...
    """
    plt = need("matplotlib.pyplot", "the image export")
    colormaps = need("matplotlib", "the image export").colormaps

    def norm(a):
        a = a.T[::-1]
        lo, hi = np.nanmin(a), np.nanmax(a)
        return np.nan_to_num((a - lo) / (hi - lo) if hi > lo else np.zeros_like(a))

    img = np.vstack((colormaps["Blues"](norm(road.normal_map()))[..., :3],
                     colormaps["Greys"](norm(road.z))[..., :3]))
    path = os.fspath(path)
    if n_slices <= 1:
        plt.imsave(path, img)
        return
    root, ext = os.path.splitext(path)
    w = img.shape[1] // n_slices
    for i in range(n_slices):
        plt.imsave(f"{root}{i}{ext or '.png'}", img[:, i * w:(i + 1) * w])


_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style></head>
<body><div id="map"></div><script>
var map = L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 22, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
var ref = {ref};
var left = {left};
var right = {right};
L.polyline(left, {{color: 'gray', weight: 1}}).addTo(map);
L.polyline(right, {{color: 'gray', weight: 1}}).addTo(map);
var line = L.polyline(ref, {{color: 'red', weight: 3}}).addTo(map);
L.marker(ref[0]).addTo(map).bindPopup('start (u = {u0})');
L.marker(ref[ref.length - 1]).addTo(map).bindPopup('end (u = {u1})');
map.fitBounds(line.getBounds());
</script></body></html>
"""


def to_html_map(road, path, n_points=2000):
    """Write a web page showing the reference line and the road edges on OpenStreetMap.

    The road needs WGS84 information (map projection, or start and end
    latitude / longitude).
    """
    idx = np.unique(np.linspace(0, road.nu - 1, min(n_points, road.nu)).astype(int))
    u = road.u[idx]

    def coords(v):
        x, y = road.uv_to_xy(u, np.full(u.size, v))
        lat, lon = road.xy_to_wgs84(x, y)
        return "[" + ",".join(f"[{a:.8f},{b:.8f}]" for a, b in zip(np.atleast_1d(lat), np.atleast_1d(lon))) + "]"

    title = os.path.basename(road.filename) if road.filename else "OpenCRG road"
    html = _HTML.format(title=title, ref=coords(0.0), left=coords(road.v_max), right=coords(road.v_min),
                        u0=f"{road.u_start:g}", u1=f"{road.u_end:g}")
    with open(os.fspath(path), "w", encoding="utf-8") as fh:
        fh.write(html)

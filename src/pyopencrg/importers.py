"""
Creation of roads from measured data: point clouds and STL meshes.

Example: convert a scanned STL mesh of a test track to OpenCRG::

    import pyopencrg as crg
    road = crg.importers.from_stl("track.stl", u_inc=0.005)   # straight track
    road.save("track.crg")

For a curved track, give the center line (points in driving order)::

    road = crg.importers.from_stl("track.stl", u_inc=0.01, reference_line=(xc, yc))
"""

from __future__ import annotations

import os

import numpy as np

from ._optional import need
from .generate import reference_line_from_points
from .road import CrgRoad

__all__ = ["read_stl", "from_point_cloud", "from_stl"]


def read_stl(path):
    """Read an STL mesh (binary or text).

    Returns
    -------
    vertices : numpy.ndarray (n, 3)
        Unique vertices.
    faces : numpy.ndarray (m, 3)
        Vertex indices of each triangle.
    """
    path = os.fspath(path)
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(84)
    n_tri = int(np.frombuffer(head[80:84], dtype="<u4")[0]) if len(head) >= 84 else -1
    if n_tri >= 0 and size == 84 + 50 * n_tri:
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        tri = np.fromfile(path, dtype=dt, offset=84, count=n_tri)["v"].astype(np.float64)
    else:
        with open(path, encoding="latin-1") as fh:
            vals = [line.split()[1:4] for line in fh if line.lstrip().startswith("vertex")]
        tri = np.asarray(vals, dtype=np.float64).reshape(-1, 3, 3)
    verts, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3)


def _auto_reference_line(x, y, u_inc):
    """Straight line along the main direction of the points (principal axis)."""
    c = np.array([x.mean(), y.mean()])
    pts = np.column_stack((x, y)) - c
    _, _, vt = np.linalg.svd(pts[:: max(1, pts.shape[0] // 200000)], full_matrices=False)
    d = vt[0]
    if d[0] < 0:
        d = -d
    s = pts @ d
    n = int(np.floor((s.max() - s.min()) / u_inc)) + 1
    t = s.min() + np.arange(n) * u_inc
    return c[0] + t * d[0], c[1] + t * d[1]


def from_point_cloud(x, y, z, u_inc=0.01, v_inc=None, reference_line=None, width=None, smoothing=None,
                     extend: float = 0.0, outside: str = "nearest", zero_start: bool = True):
    """Build a road from a cloud of measured points (x, y, z).

    Parameters
    ----------
    x, y, z : array_like
        Measured points.
    u_inc, v_inc : float
        Grid increments of the result (m). ``v_inc`` defaults to ``u_inc``.
    reference_line : (array, array), optional
        Points (x, y) of the road center line in driving order. Default: a
        straight line along the main direction of the cloud.
    width : float or (float, float), optional
        Width of the road (m) or (right, left) edges. Default: the width
        covered by the points.
    smoothing : float, optional
        Smoothing of the reference line (see :func:`pyopencrg.generate.reference_line_from_points`).
    extend : float
        Length (m) added straight before and after the reference line.
    outside : str
        What to do for grid points outside the cloud: ``"nearest"`` (value of
        the closest point) or ``"nan"``.
    zero_start : bool
        Shift the heights so that the start of the reference line is at z = 0.

    Returns
    -------
    CrgRoad
    """
    itp = need("scipy.interpolate", "from_point_cloud()")
    LinearNDInterpolator, NearestNDInterpolator = itp.LinearNDInterpolator, itp.NearestNDInterpolator

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    v_inc = u_inc if v_inc is None else v_inc
    if reference_line is None:
        xr, yr = _auto_reference_line(x, y, u_inc)
    else:
        xr, yr = reference_line_from_points(reference_line[0], reference_line[1], u_inc, smoothing)
    if extend > 0:
        n = int(round(extend / u_inc))
        k = np.arange(n, 0, -1) * u_inc
        p0 = np.arctan2(yr[1] - yr[0], xr[1] - xr[0])
        p1 = np.arctan2(yr[-1] - yr[-2], xr[-1] - xr[-2])
        xr = np.concatenate((xr[0] - k * np.cos(p0), xr, xr[-1] + k[::-1] * np.cos(p1)))
        yr = np.concatenate((yr[0] - k * np.sin(p0), yr, yr[-1] + k[::-1] * np.sin(p1)))
    phi = np.arctan2(np.diff(yr), np.diff(xr))
    probe = CrgRoad(np.zeros((xr.size, 2)), u_inc, [-1.0, 1.0], heading=phi, x_start=xr[0], y_start=yr[0])
    if width is None:
        step = max(1, x.size // 500000)
        _, vp = probe.xy_to_uv(x[::step], y[::step])
        lo = np.floor(np.min(vp) / v_inc) * v_inc
        hi = np.ceil(np.max(vp) / v_inc) * v_inc
    else:
        w = np.atleast_1d(np.asarray(width, dtype=np.float64))
        lo, hi = (-w[0] / 2, w[0] / 2) if w.size == 1 else (w[0], w[1])
    nv = int(round((hi - lo) / v_inc)) + 1
    v = lo + np.arange(nv) * v_inc
    road = CrgRoad(np.zeros((xr.size, nv)), u_inc, v, heading=phi, x_start=xr[0], y_start=yr[0],
                   comments=["road generated by pyopencrg from a point cloud"])
    gx, gy = road.grid_xy()
    pts = np.column_stack((x, y))
    zi = LinearNDInterpolator(pts, z)(gx, gy)
    if outside == "nearest":
        miss = np.isnan(zi)
        if miss.any():
            zi[miss] = NearestNDInterpolator(pts, z)(gx[miss], gy[miss])
    elif outside != "nan":
        raise ValueError("outside must be 'nearest' or 'nan'")
    if zero_start:
        ref = road.uv_to_index(road.u_start, 0.0)[1]
        row = zi[0]
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size:
            zi -= row[valid[np.argmin(np.abs(valid - ref))]]
    road.z = zi
    return road


def from_stl(path, u_inc=0.01, v_inc=None, **kwargs):
    """Build a road from an STL mesh (e.g. a laser scan).

    All vertices of the mesh are used as a point cloud; see
    :func:`from_point_cloud` for the other parameters.
    """
    verts, _ = read_stl(path)
    road = from_point_cloud(verts[:, 0], verts[:, 1], verts[:, 2], u_inc, v_inc, **kwargs)
    road.comments = [f"road converted by pyopencrg from {os.path.basename(os.fspath(path))}"]
    return road

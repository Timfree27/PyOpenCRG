"""
Analysis tools: statistics, normal map, scanning artifacts, holes, peaks.

These functions help to find and repair defects of measured roads (laser
scans): very steep points (artifacts), holes (NaN values) and isolated peaks.

Typical cleaning workflow::

    road = crg.read("scan.crg")
    plt.hist(road.normal_map().ravel(), bins=100, log=True)   # choose a tolerance
    cleaned = road.remove_artifacts(normal_tol=0.3, radius=15).fill_holes("linear")
    cleaned.save("scan_clean.crg")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["stats", "normal_map", "artifact_mask", "remove_artifacts", "nan_regions", "fill_holes",
           "find_peaks", "PeakResult"]


def stats(road) -> dict:
    """Statistics of the elevation grid.

    Returns
    -------
    dict
        ``min``, ``max``, ``mean``, ``std``, ``rms`` of the grid values (m),
        ``nan_ratio`` (0 to 1) and ``nan_count``.
    """
    z = road.z
    fin = np.isfinite(z)
    out = {"nan_count": int((~fin).sum()), "nan_ratio": float(1 - fin.mean())}
    if fin.any():
        zz = z[fin]
        out.update(min=float(zz.min()), max=float(zz.max()), mean=float(zz.mean()),
                   std=float(zz.std()), rms=float(np.sqrt(np.mean(zz ** 2))))
    else:
        out.update(min=np.nan, max=np.nan, mean=np.nan, std=np.nan, rms=np.nan)
    return out


def _spacing_v(road):
    return road.v_inc if road.v_inc is not None else float(np.mean(np.diff(road.v)))


def normal_map(road, absolute: bool = False) -> np.ndarray:
    """Vertical component of the unit normal vector of the surface at every node.

    1 means a flat surface, values close to 0 a vertical wall. Laser scan
    artifacts usually show up as very low values.

    Parameters
    ----------
    absolute : bool
        Use the absolute surface (with slope and banking) instead of the grid.

    Returns
    -------
    numpy.ndarray of shape (nu, nv)
    """
    z = road.surface_z() if absolute else road.z
    if road.v_inc is not None:
        gu, gv = np.gradient(z, road.u_inc, road.v_inc)
    else:
        gu, gv = np.gradient(z, road.u_inc, road.v)
    return 1.0 / np.sqrt(1.0 + gu ** 2 + gv ** 2)


def artifact_mask(road, normal_tol: float = 0.3, radius: float = 15) -> np.ndarray:
    """Boolean mask of the artifacts and of their surroundings.

    Parameters
    ----------
    normal_tol : float
        Nodes whose normal vertical component is below this value are artifacts.
    radius : float
        Distance around the artifacts (in grid points) also included.
    """
    from scipy.ndimage import distance_transform_edt

    bad = normal_map(road) < normal_tol
    if not bad.any() or radius <= 0:
        return bad
    return distance_transform_edt(~bad) <= radius


def remove_artifacts(road, normal_tol: float = 0.3, radius: float = 15):
    """Return a copy of the road where the artifacts (see :func:`artifact_mask`) are NaN."""
    z = road.z.copy()
    z[artifact_mask(road, normal_tol, radius)] = np.nan
    return road._derive(z=z)


def _merge_boxes(boxes):
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                b = boxes[i]
                if a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]:
                    a = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return sorted(tuple(b) for b in boxes)


def nan_regions(z, margin: int = 50):
    """Rectangles around the connected groups of NaN values of a grid.

    Parameters
    ----------
    z : numpy.ndarray or CrgRoad
        Grid (or road) to analyse.
    margin : int
        Number of grid points added around each group. Overlapping
        rectangles are merged.

    Returns
    -------
    list of (iu0, iu1, iv0, iv1)
        Index ranges (both ends included).
    """
    from scipy.ndimage import find_objects, label

    if hasattr(z, "z"):
        z = z.z
    nan = np.isnan(z)
    if not nan.any():
        return []
    lab, _ = label(nan)
    nu, nv = z.shape
    boxes = []
    for sl in find_objects(lab):
        if sl is None:
            continue
        boxes.append((max(0, sl[0].start - margin), min(nu - 1, sl[0].stop - 1 + margin),
                      max(0, sl[1].start - margin), min(nv - 1, sl[1].stop - 1 + margin)))
    return _merge_boxes(boxes)


def _fill_patch(patch, method, max_points, du, dv):
    from scipy.interpolate import griddata

    nan = np.isnan(patch)
    if nan.all():
        return patch
    ii, jj = np.nonzero(~nan)
    pts = np.column_stack((ii * du, jj * dv))
    vals = patch[~nan]
    qi, qj = np.nonzero(nan)
    q = np.column_stack((qi * du, qj * dv))
    out = patch.copy()
    if method == "rbf":
        from scipy.interpolate import RBFInterpolator
        if pts.shape[0] > max_points:
            step = int(np.ceil(pts.shape[0] / max_points))
            pts, vals = pts[::step], vals[::step]
        res = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=1e-9, neighbors=min(64, pts.shape[0]))(q)
    else:
        res = griddata(pts, vals, q, method=method)
        miss = np.isnan(res)
        if miss.any():  # outside of the convex hull
            res[miss] = griddata(pts, vals, q[miss], method="nearest")
    out[qi, qj] = res
    return out


def fill_holes(road, method: str = "linear", margin: int = 5, max_points: int = 40000):
    """Fill the NaN holes of the grid by 2D interpolation.

    Each hole is interpolated from the valid values around it only, which is
    much faster than interpolating the whole grid.

    Parameters
    ----------
    method : str
        ``"linear"``, ``"cubic"``, ``"nearest"`` or ``"rbf"``.
    margin : int
        Number of valid points around each hole used for the interpolation.
    max_points : int
        Maximum number of known points for the ``"rbf"`` method.

    Returns
    -------
    CrgRoad
    """
    if method not in ("linear", "cubic", "nearest", "rbf"):
        raise ValueError("method must be 'linear', 'cubic', 'nearest' or 'rbf'")
    z = road.z.copy()
    du, dv = road.u_inc, _spacing_v(road)
    for iu0, iu1, iv0, iv1 in nan_regions(z, margin):
        z[iu0:iu1 + 1, iv0:iv1 + 1] = _fill_patch(z[iu0:iu1 + 1, iv0:iv1 + 1], method, max_points, du, dv)
    return road._derive(z=z)


@dataclass
class PeakResult:
    """Result of :func:`find_peaks`.

    Attributes
    ----------
    iu, iv : numpy.ndarray
        Grid indices of the peak points (0 based).
    u, v : numpy.ndarray
        Road coordinates of the peak points (m).
    z : numpy.ndarray
        Grid elevation of the peak points (m).
    """

    iu: np.ndarray
    iv: np.ndarray
    u: np.ndarray
    v: np.ndarray
    z: np.ndarray

    def __len__(self):
        return self.iu.size


def find_peaks(road, threshold: float = 0.5, size: int = 3, iu=None, iv=None) -> PeakResult:
    """Detect peaks (isolated high or low points) in the elevation grid.

    A Laplacian edge detector first selects candidate points, then the
    points of the surrounding window whose height is clearly above the
    local median are kept.

    Parameters
    ----------
    threshold : float
        Threshold of the edge detector (larger = fewer peaks).
    size : int
        Half size of the search window (grid points); should be larger than
        the expected peak size.
    iu, iv : (int, int), optional
        Restrict the search to this part of the grid.

    Returns
    -------
    PeakResult
    """
    from scipy.ndimage import median_filter, minimum_filter
    from scipy.signal import correlate2d

    from .edit import _index_range

    i0, i1 = _index_range(iu, road.nu, "u")
    j0, j1 = _index_range(iv, road.nv, "v")
    z = road.z[i0:i1 + 1, j0:j1 + 1]
    az = np.abs(np.nan_to_num(z))
    lap = np.full((3, 3), -1.0)
    lap[1, 1] = 8.0
    edges = np.zeros_like(az)
    if az.shape[0] > 2 and az.shape[1] > 2:
        edges[1:-1, 1:-1] = correlate2d(az, lap * 10.0, mode="valid")
    cand = edges > threshold
    ra = int(size)
    nr, nc = az.shape
    win = 2 * ra + 1
    mask = np.zeros_like(cand)
    if cand.any() and nr > win and nc > win:
        med = np.maximum(median_filter(np.nan_to_num(z), size=win, mode="nearest"), 0.05)
        ci, cj = np.nonzero(cand)
        ci = np.clip(ci, ra, nr - ra - 1)
        cj = np.clip(cj, ra, nc - ra - 1)
        thr = np.full(az.shape, np.inf)
        np.minimum.at(thr, (ci, cj), 2.0 * med[ci, cj])
        mask = az > minimum_filter(thr, size=win, mode="constant", cval=np.inf)
    pi, pj = np.nonzero(mask)
    pi_g, pj_g = pi + i0, pj + j0
    return PeakResult(iu=pi_g, iv=pj_g, u=road.u[pi_g], v=road.v[pj_g], z=road.z[pi_g, pj_g])

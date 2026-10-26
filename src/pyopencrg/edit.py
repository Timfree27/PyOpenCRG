"""
Editing functions: cut, append, flip, filter, re-sample, slope and banking...

All functions take a :class:`pyopencrg.CrgRoad` and return a **new** road;
the input road is never modified. They are also available as methods of the
road, which is the simplest way to use them::

    part = road.crop(u=(0, 100))       # same as pyopencrg.edit.crop(road, (0, 100))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .keywords import DEFAULT_MODIFIERS
from .road import integrate_heading, integrate_slope

__all__ = [
    "crop", "crop_index", "trim", "trim_nan_borders", "add_ramps", "append", "concatenate", "flip",
    "filter_grid", "clip_z", "resample", "set_banking", "set_reference_elevation", "set_slope",
    "extract_slope", "extract_banking", "separate_slope_banking", "overlay_uv", "overlay_xy",
    "fill_nan", "add_profile", "add_random", "engrave_image", "compare", "ComparisonResult",
]


def _stamp(road, text):
    road.comments = list(road.comments) + [f"* {text} by pyopencrg at {datetime.now():%Y-%m-%d %H:%M:%S}"]


def _index_range(rng, n, name):
    if rng is None:
        return 0, n - 1
    a, b = int(rng[0]), int(rng[1])
    a = a + n if a < 0 else a
    b = b + n if b < 0 else b
    if not 0 <= a < b <= n - 1:
        raise ValueError(f"illegal {name} index range {tuple(rng)} for a grid of {n} values")
    return a, b


def _keep_mods(road, new):
    mods = road.mods
    if mods is None or not mods or mods == DEFAULT_MODIFIERS:
        new.mods = None if mods is None else dict(mods)
    else:
        new.mods = {}
        warnings.warn("non default modifiers were removed (they would not fit the modified road)", stacklevel=3)


def _update_wgs84(road, new, i_start, i_end):
    """Update WGS84 header data after the reference line was cut or extended."""
    if road.alt_start is not None:
        new.alt_start = road.alt_start + (new.zr[0] - road.zr[0])
        new.alt_end = new.alt_start + (new.zr[-1] - new.zr[0])
    if road.mpro or road.lon_start is None:
        return
    if road.lon_end is not None:
        from .wgs84 import road_xy_to_wgs84
        try:
            lat, lon = road_xy_to_wgs84(road, np.array([new.x[0], new.x[-1]]), np.array([new.y[0], new.y[-1]]))
        except ValueError as exc:  # inconsistent WGS84 data in the source road
            new.lon_start = new.lat_start = new.lon_end = new.lat_end = None
            warnings.warn(f"WGS84 positions removed: {exc}", stacklevel=3)
            return
        new.lat_start, new.lon_start = float(lat[0]), float(lon[0])
        new.lat_end, new.lon_end = float(lat[1]), float(lon[1])
    elif i_start != 0:
        new.lon_start = new.lat_start = None
        warnings.warn("WGS84 start position removed (cannot be recomputed without an end position)", stacklevel=3)


# ---------------------------------------------------------------------------
# cutting and extending
# ---------------------------------------------------------------------------

def crop_index(road, iu=None, iv=None):
    """Keep the part of the grid between indices ``iu = (first, last)`` and ``iv = (first, last)``.

    Indices are 0 based and both ends are included. Negative indices count
    from the end (``-1`` = last).
    """
    nu, nv = road.shape
    i0, i1 = _index_range(iu, nu, "u")
    j0, j1 = _index_range(iv, nv, "v")
    new = road._derive(z=road.z[i0:i1 + 1, j0:j1 + 1].copy(), v=road.v[j0:j1 + 1].copy(),
                       u_start=road.u_start + i0 * road.u_inc, x=road.x[i0:i1 + 1].copy(),
                       y=road.y[i0:i1 + 1].copy(), zr=road.zr[i0:i1 + 1].copy(),
                       banking=road.banking[i0:i1 + 1].copy())
    _update_wgs84(road, new, i0, i1)
    if (i0, i1, j0, j1) != (0, nu - 1, 0, nv - 1):
        _keep_mods(road, new)
        _stamp(new, f"cut to iu=[{i0} {i1}] iv=[{j0} {j1}]")
    return new


def crop(road, u=None, v=None):
    """Keep the part of the road between ``u = (u_first, u_last)`` and ``v = (v_right, v_left)`` (metres)."""
    eps = 1e-9
    iu = iv = None
    if u is not None:
        lo = road.u_start if u[0] is None else u[0]
        hi = road.u_end if u[1] is None else u[1]
        i0 = int(np.ceil((lo - road.u_start) / road.u_inc - eps))
        i1 = int(np.floor((hi - road.u_start) / road.u_inc + eps))
        iu = (max(i0, 0), min(i1, road.nu - 1))
    if v is not None:
        lo = road.v_min if v[0] is None else v[0]
        hi = road.v_max if v[1] is None else v[1]
        idx = np.flatnonzero((road.v >= lo - eps) & (road.v <= hi + eps))
        if idx.size < 2:
            raise ValueError(f"less than 2 long sections between v = {lo} and {hi}")
        iv = (int(idx[0]), int(idx[-1]))
    return crop_index(road, iu, iv)


def trim(road, before=0.0, after=0.0):
    """Remove ``before`` metres at the start and ``after`` metres at the end of the road."""
    if before < 0 or after < 0:
        raise ValueError("before and after must be positive")
    i0 = int(round(before / road.u_inc))
    i1 = road.nu - 1 - int(round(after / road.u_inc))
    if i0 >= i1:
        raise ValueError("the cut is longer than the road")
    return crop_index(road, (i0, i1))


def trim_nan_borders(road):
    """Remove the cross sections at both ends of the road that contain NaN values."""
    bad = np.isnan(road.z).any(axis=1)
    good = np.flatnonzero(~bad)
    if good.size < 2:
        raise ValueError("less than 2 cross sections without NaN")
    if good[0] == 0 and good[-1] == road.nu - 1:
        return road.copy()
    return crop_index(road, (good[0], good[-1]))


def add_ramps(road, before=2.0, after=2.0):
    """Extend the road with smooth access ramps before the start and after the end.

    The added cross sections go linearly from the mean height of the first
    (last) cross section to its real shape. The reference line is extended
    straight, with the start (end) heading and slope.
    """
    if before < 0 or after < 0:
        raise ValueError("before and after must be positive")
    inc = road.u_inc
    nb, na = int(round(before / inc)), int(round(after / inc))
    if nb == 0 and na == 0:
        return road.copy()
    z = road.z
    first, last = z[0], z[-1]
    a = (np.arange(nb) + 1) / (nb + 1)
    zb = (1 - a)[:, None] * np.nanmean(first) + a[:, None] * first[None, :]
    a = (np.arange(na) + 1) / (na + 1)
    za = (1 - a)[:, None] * last[None, :] + a[:, None] * np.nanmean(last)
    kb = np.arange(nb, 0, -1) * inc
    ka = np.arange(1, na + 1) * inc
    cb, sb = np.cos(road.phi_start), np.sin(road.phi_start)
    ce, se = np.cos(road.phi_end), np.sin(road.phi_end)
    s0, s1 = road.slope[0], road.slope[-1]
    x = np.concatenate((road.x[0] - kb * cb, road.x, road.x[-1] + ka * ce))
    y = np.concatenate((road.y[0] - kb * sb, road.y, road.y[-1] + ka * se))
    zr = np.concatenate((road.zr[0] - kb * s0, road.zr, road.zr[-1] + ka * s1))
    b = np.concatenate((np.full(nb, road.banking[0]), road.banking, np.full(na, road.banking[-1])))
    new = road._derive(z=np.vstack((zb, z, za)), u_start=road.u_start - nb * inc, x=x, y=y, zr=zr, banking=b)
    _update_wgs84(road, new, -1, -1)
    _keep_mods(road, new)
    _stamp(new, f"ramps of {nb * inc:g} m / {na * inc:g} m added")
    return new


def append(road1, road2):
    """Append ``road2`` at the end of ``road1``.

    ``road2`` is moved and rotated so that its first interval overlaps the
    last interval of ``road1`` (the last cross section of ``road1`` and the
    first one of ``road2`` are dropped). Both roads must have the same
    increment along u and the same lateral positions v.

    Returns
    -------
    CrgRoad
        The combined road. Its length is ``road1.length + road2.length - u_inc``.
    """
    from .mods import transform

    inc = road1.u_inc
    eps = road1.get_option("check_eps")
    if abs(inc - road2.u_inc) > eps * (inc + road2.u_inc):
        raise ValueError(f"increments along u differ: {road1.u_inc} and {road2.u_inc}")
    if road1.nv != road2.nv or np.max(np.abs(road1.v - road2.v)) > eps * max(np.max(np.abs(road1.v)), 1.0):
        raise ValueError("the lateral positions v of both roads must be identical")
    z2 = road2.z.copy()
    zr2 = road2.zr.copy()
    # make the slope of the first interval of road2 equal to the last slope of road1
    sdel = road1.slope[-1] - road2.slope[0]
    if abs(sdel) > eps:
        zr2[0] -= sdel * inc
        z2[0] += sdel * inc
    r2 = road2._derive(z=z2, zr=zr2)
    dphi = road1.phi_end - r2.phi_start
    moved = transform(r2, 0.0, 0.0, 0.0, dphi)
    moved = transform(moved, road1.x[-2] - moved.x[0], road1.y[-2] - moved.y[0], road1.zr[-2] - moved.zr[0])
    new = road1._derive(
        z=np.vstack((road1.z[:-1], moved.z[1:])),
        x=np.concatenate((road1.x[:-1], moved.x[1:])),
        y=np.concatenate((road1.y[:-1], moved.y[1:])),
        zr=np.concatenate((road1.zr[:-1], moved.zr[1:])),
        banking=np.concatenate((road1.banking[:-1], moved.banking[1:])),
    )
    new.lon_end = new.lat_end = None
    if road1.lon_end is not None and road2.lon_start is not None and road2.lon_end is not None and not road1.mpro:
        from .wgs84 import distance, road_xy_to_wgs84
        lat, lon = road_xy_to_wgs84(road1, road1.x[-2], road1.y[-2])
        d = distance(lat, lon, road2.lat_start, road2.lon_start)[0]
        if d < eps * inc * new.nu + 0.01:
            new.lat_end, new.lon_end = road2.lat_end, road2.lon_end
        else:
            warnings.warn("WGS84 positions of both roads do not match: WGS84 end position removed", stacklevel=2)
    if road1.alt_start is not None:
        new.alt_end = road1.alt_start + (new.zr[-1] - new.zr[0])
    _stamp(new, "roads appended")
    return new


def concatenate(roads):
    """Append several roads one after the other (see :func:`append`)."""
    roads = list(roads)
    if not roads:
        raise ValueError("no road given")
    out = roads[0]
    for r in roads[1:]:
        out = append(out, r)
    return out


def flip(road):
    """Reverse the driving direction of a road (start <-> end, left <-> right)."""
    new = road._derive(z=road.z[::-1, ::-1].copy(), v=-road.v[::-1], x=road.x[::-1].copy(),
                       y=road.y[::-1].copy(), zr=road.zr[::-1].copy(), banking=-road.banking[::-1])
    new.lon_start, new.lat_start, new.alt_start = road.lon_end, road.lat_end, road.alt_end
    new.lon_end, new.lat_end, new.alt_end = road.lon_start, road.lat_start, road.alt_start
    if new.lon_start is None:
        new.lon_end = new.lat_end = None
    if new.alt_start is None:
        new.alt_end = None
    _stamp(new, "flipped")
    return new


# ---------------------------------------------------------------------------
# filtering and limiting
# ---------------------------------------------------------------------------

def _binomial(n, k):
    m = np.array([1.0])
    for _ in range(n - k - 1):
        m = np.convolve(m, [1.0, 1.0])
    for _ in range(k):
        m = np.convolve(m, [-1.0, 1.0])
    return m * 2.0 ** (-n + k + 1)


def filter_mask(method, size):
    """Convolution mask used by :func:`filter_grid` (rows along u, columns along v)."""
    nu_, nv_ = (size, size) if np.isscalar(size) else (int(size[0]), int(size[1]))
    method = method.lower()
    if method == "mean":
        return np.full((nu_, nv_), 1.0 / (nu_ * nv_))
    ku, kv = {"gauss": (0, 0), "sobel": (1, 0), "2diff": (2, 0), "laplace": (2, 2)}.get(method, (None, None))
    if ku is None:
        raise ValueError("method must be 'mean', 'gauss', 'sobel', '2diff' or 'laplace'")
    if nu_ <= ku or nv_ <= kv:
        raise ValueError("the mask is too small for this derivative")
    mask = np.outer(_binomial(nu_, ku), _binomial(nv_, kv))
    if method != "gauss" and abs(mask.sum()) < 1e-12:
        mask[(nu_ - 1) // 2, (nv_ - 1) // 2] += 1.0
    return mask


def filter_grid(road, method="mean", size=(3, 3), weight=1.0, repeat=1, iu=None, iv=None):
    """Filter the elevation grid with a convolution mask.

    Only the grid values are filtered (the reference line elevation and the
    banking are not changed). The border of the selected area, where the
    mask does not fit, is left unchanged.

    See :meth:`pyopencrg.CrgRoad.filter` for the parameters.
    """
    from scipy.signal import convolve2d

    i0, i1 = _index_range(iu, road.nu, "u")
    j0, j1 = _index_range(iv, road.nv, "v")
    mask = filter_mask(method, size) * weight
    z = road.z.copy()
    sub = z[i0:i1 + 1, j0:j1 + 1]
    r0, c0 = (mask.shape[0] - 1) // 2, (mask.shape[1] - 1) // 2
    for _ in range(int(repeat)):
        tz = convolve2d(sub, mask, mode="valid")
        sub[r0:r0 + tz.shape[0], c0:c0 + tz.shape[1]] = tz
    return road._derive(z=z)


def clip_z(road, z_min=None, z_max=None, iu=None, iv=None, absolute=True):
    """Limit the elevation of a part of the road between ``z_min`` and ``z_max``."""
    i0, i1 = _index_range(iu, road.nu, "u")
    j0, j1 = _index_range(iv, road.nv, "v")
    z = road.z.copy()
    sub = z[i0:i1 + 1, j0:j1 + 1]
    if absolute:
        contrib = road.zr[i0:i1 + 1, None] + road.banking[i0:i1 + 1, None] * road.v[None, j0:j1 + 1]
        sub[...] = np.clip(sub + contrib, z_min, z_max) - contrib
    else:
        sub[...] = np.clip(sub, z_min, z_max)
    return road._derive(z=z)


# ---------------------------------------------------------------------------
# re-sampling
# ---------------------------------------------------------------------------

def _new_v(road, v_inc, v):
    inc = v_inc if v_inc is not None else (road.v_inc if road.v_inc is not None else 0.01)
    if v is None:
        if v_inc is None:
            return road.v.copy()
        lo, hi = road.v_min, road.v_max
    else:
        va = np.atleast_1d(np.asarray(v, dtype=np.float64))
        if va.size == 1:
            if va[0] <= 0:
                raise ValueError("the half width must be positive")
            lo, hi = -va[0], va[0]
        elif va.size == 2:
            lo, hi = va
        else:
            return np.sort(va)
    n = int(np.floor((hi - lo) / inc + 1e-9)) + 1
    return lo + np.arange(n) * inc


def resample(road, u_inc=None, v_inc=None, v=None):
    """Re-sample a road on a new grid (see :meth:`pyopencrg.CrgRoad.resample`)."""
    from scipy.interpolate import CubicSpline

    inc = road.u_inc if u_inc is None else float(u_inc)
    if inc <= 0:
        raise ValueError("u_inc must be positive")
    nu = int(np.floor(road.length / inc + 1e-9)) + 1
    u_new = road.u_start + np.arange(nu) * inc
    if road.is_straight:
        x, y = integrate_heading(road.x[0], road.y[0], np.full(nu - 1, road.phi_start), inc)
    else:
        u_old = road.u
        sx, sy = CubicSpline(u_old, road.x), CubicSpline(u_old, road.y)
        phi = np.arctan2(np.diff(sy(u_new)), np.diff(sx(u_new)))
        x, y = integrate_heading(road.x[0], road.y[0], phi, inc)
    zr = np.interp(u_new, road.u, road.zr)
    b = np.interp(u_new, road.u, road.banking)
    v_new = _new_v(road, v_inc, v)
    uu, vv = np.meshgrid(u_new, v_new, indexing="ij")
    z = road.uv_to_z(uu, vv, grid_only=True)
    new = road._derive(z=z, v=v_new, u_inc=inc, x=x, y=y, zr=zr, banking=b)
    _stamp(new, f"re-sampled to u_inc={inc:g}")
    return new


# ---------------------------------------------------------------------------
# slope and banking
# ---------------------------------------------------------------------------

def set_banking(road, banking=0.0):
    """Change the banking while keeping the same road surface."""
    b = np.broadcast_to(np.asarray(banking, dtype=np.float64), (road.nu,)).astype(np.float64)
    z = road.z + (road.banking - b)[:, None] * road.v[None, :]
    return road._derive(z=z, banking=b)


def set_reference_elevation(road, zr=None):
    """Change the reference line elevation while keeping the same road surface."""
    if zr is None:
        zr = 0.5 * (road.zr[0] + road.zr[-1])
    zr = np.atleast_1d(np.asarray(zr, dtype=np.float64))
    if zr.size == 1:
        zr = np.full(road.nu, zr[0])
    elif zr.size == 2:
        zr = np.linspace(zr[0], zr[1], road.nu)
    elif zr.size != road.nu:
        raise ValueError(f"zr must have 1, 2 or nu={road.nu} values")
    new = road._derive(z=road.z + (road.zr - zr)[:, None], zr=zr.copy())
    if road.alt_start is not None:
        new.alt_start = road.alt_start + zr[0] - road.zr[0]
        new.alt_end = new.alt_start + zr[-1] - zr[0]
    return new


def set_slope(road, slope):
    """Change the longitudinal slope (one value or nu - 1 values) while keeping the same surface."""
    s = np.broadcast_to(np.asarray(slope, dtype=np.float64), (road.nu - 1,))
    return set_reference_elevation(road, integrate_slope(road.zr[0], s, road.u_inc))


def extract_slope(road, smoothing=0.01):
    """Fit a smooth longitudinal profile on the road center and put it in the reference line."""
    from .smoothing import smoothing_spline

    u = road.u
    prof = road.uv_to_z(u, np.zeros(road.nu))
    ok = np.isfinite(prof)
    if ok.sum() < 2:
        raise ValueError("not enough valid values along v = 0")
    fit = smoothing_spline(u[ok], prof[ok], smoothing)(u)
    return set_reference_elevation(road, road.zr[0] + fit - fit[0])


def _lateral_regression(z, v):
    """Least squares line z = c + b * v fitted on every cross section (NaN ignored)."""
    ok = np.isfinite(z)
    w = ok.astype(np.float64)
    zz = np.where(ok, z, 0.0)
    n = w.sum(axis=1)
    sv = (w * v).sum(axis=1)
    sz = zz.sum(axis=1)
    svv = (w * v * v).sum(axis=1)
    svz = (zz * v).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * svv - sv * sv
        b = (n * svz - sv * sz) / den
        c = (sz - b * sv) / n
    bad = ~np.isfinite(b)
    if bad.any():
        if bad.all():
            raise ValueError("no cross section has 2 valid values")
        idx = np.arange(b.size)
        b[bad] = np.interp(idx[bad], idx[~bad], b[~bad])
        c[bad] = np.interp(idx[bad], idx[~bad], c[~bad])
    return c, b


def extract_banking(road, smoothing=0.01):
    """Fit the banking of each cross section, smooth it along u and put it in the reference line."""
    from .smoothing import smoothing_spline

    _, b = _lateral_regression(road.z + road.banking[:, None] * road.v[None, :], road.v)
    if abs(smoothing - 1.0) > 1e-12:
        idx = np.arange(road.nu, dtype=np.float64)
        b = smoothing_spline(idx, b, smoothing)(idx)
    return set_banking(road, b)


def separate_slope_banking(road, slope_window=0.0, banking_window=0.0):
    """Find, filter and separate slope and banking (see :meth:`pyopencrg.CrgRoad.separate_slope_banking`)."""
    from .smoothing import fir_smooth

    total = road.surface_z()
    c, b = _lateral_regression(total, road.v)
    nu = road.nu
    sw = slope_window
    if sw > 0:
        w = int(round(sw / road.u_inc / 2))
        if 0 < w < nu:
            zf = fir_smooth(c, w, window="hann", reflect="point")
        else:
            zf = np.linspace(c[0], c[-1], nu)
    elif sw == 0:
        zf = np.full(nu, c.mean())
    elif sw == -1:
        zf = c
    elif sw == -2:
        zf = np.linspace(c[0], c[-1], nu)
    else:
        raise ValueError("illegal slope_window value")
    new = set_reference_elevation(road, zf)
    bw = banking_window
    if bw > 0:
        w = int(round(bw / road.u_inc / 2))
        bf = fir_smooth(b, w, window="hann", reflect="point") if 0 < w < nu else np.full(nu, b.mean())
    elif bw == 0:
        bf = np.zeros(nu)
    elif bw == -1:
        bf = b
    elif bw == -2:
        bf = np.full(nu, b.mean())
    else:
        raise ValueError("illegal banking_window value")
    return set_banking(new, bf)


# ---------------------------------------------------------------------------
# combining roads
# ---------------------------------------------------------------------------

def overlay_uv(road, other, iu=None, iv=None):
    """Add the surface of ``other`` to ``road``, matched in road coordinates (u, v).

    ``other`` is sampled with the increments of ``road`` starting at its own
    first point, and repeated if it is smaller than the selected area.
    """
    i0, i1 = _index_range(iu, road.nu, "u")
    j0, j1 = _index_range(iv, road.nv, "v")
    nu_r, nv_r = i1 - i0 + 1, j1 - j0 + 1
    ncu = int(np.floor(other.length / road.u_inc + 1e-9)) + 1
    cu = other.u_start + np.arange(ncu) * road.u_inc
    if road.v_inc is not None:
        ncv = int(np.floor(other.width / road.v_inc + 1e-9)) + 1
        cv = other.v_min + np.arange(ncv) * road.v_inc
    else:
        cv = road.v[j0:j1 + 1]
    uu, vv = np.meshgrid(cu, cv, indexing="ij")
    cz = other.uv_to_z(uu, vv)
    reps = (-(-nu_r // cz.shape[0]), -(-nv_r // cz.shape[1]))
    cz = np.tile(cz, reps)[:nu_r, :nv_r]
    z = road.z.copy()
    z[i0:i1 + 1, j0:j1 + 1] += cz
    return road._derive(z=z)


def overlay_xy(road, other, iu=None, iv=None, outside=None):
    """Add the surface of ``other`` to ``road``, matched in inertial coordinates (x, y)."""
    i0, i1 = _index_range(iu, road.nu, "u")
    j0, j1 = _index_range(iv, road.nv, "v")
    uu, vv = np.meshgrid(road.u[i0:i1 + 1], road.v[j0:j1 + 1], indexing="ij")
    x, y = road.uv_to_xy(uu, vv)
    ou, ov = other.xy_to_uv(x, y)
    cz = other.uv_to_z(ou, ov)
    if outside is not None:
        out = (ou < other.u_start) | (ou > other.u_end) | (ov < other.v_min) | (ov > other.v_max)
        cz[out] = outside
    z = road.z.copy()
    z[i0:i1 + 1, j0:j1 + 1] += cz
    return road._derive(z=z)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def fill_nan(road, mode=2, offset=0.0):
    """Replace the NaN values of the grid (see :meth:`pyopencrg.CrgRoad.fill_nan`).

    * mode 1: NaN -> ``offset``;
    * mode 2: NaN at the borders of a cross section -> last valid value +
      ``offset``; NaN between valid values -> linear interpolation along v.
      Cross sections without any valid value are copied from the closest
      valid cross section.
    """
    z = road.z.copy()
    nan = np.isnan(z)
    if not nan.any() or mode == 0:
        return road._derive(z=z)
    if mode == 1:
        z[nan] = offset
        return road._derive(z=z)
    if mode != 2:
        raise ValueError("mode must be 0, 1 or 2")
    nu, nv = z.shape
    full = nan.all(axis=1)
    if full.all():
        raise ValueError("the grid contains only NaN values")
    cols = np.arange(nv)
    prev = np.where(~nan, cols, -1)
    np.maximum.accumulate(prev, axis=1, out=prev)
    nxt = np.where(~nan, cols, nv)
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(nu)[:, None].repeat(nv, axis=1)
    v = road.v
    has_p, has_n = prev >= 0, nxt < nv
    pz = z[rows, np.clip(prev, 0, nv - 1)]
    nz = z[rows, np.clip(nxt, 0, nv - 1)]
    inner = nan & has_p & has_n
    w = (v[cols][None, :] - v[np.clip(prev, 0, nv - 1)]) / (v[np.clip(nxt, 0, nv - 1)] - v[np.clip(prev, 0, nv - 1)] + 1e-300)
    z[inner] = (pz + w * (nz - pz))[inner]
    left = nan & ~has_p & has_n  # right border (small v)
    z[left] = nz[left] + offset
    right = nan & has_p & ~has_n
    z[right] = pz[right] + offset
    if full.any():
        good = np.flatnonzero(~full)
        src = good[np.clip(np.searchsorted(good, np.flatnonzero(full)), 0, good.size - 1)]
        alt = good[np.clip(np.searchsorted(good, np.flatnonzero(full)) - 1, 0, good.size - 1)]
        bad = np.flatnonzero(full)
        pick = np.where(np.abs(src - bad) <= np.abs(alt - bad), src, alt)
        z[bad] = z[pick] + offset
    return road._derive(z=z)


# ---------------------------------------------------------------------------
# synthetic surfaces
# ---------------------------------------------------------------------------

def _profile_values(x, section, profile, name):
    s = np.asarray(section, dtype=np.float64).ravel()
    p = np.asarray(profile, dtype=np.float64).ravel()
    if s.size != p.size or s.size < 1:
        raise ValueError(f"{name} section and profile must have the same (non zero) length")
    if s.size > 1 and not (np.all(np.diff(s) >= 0) or np.all(np.diff(s) <= 0)):
        raise ValueError(f"{name} section must be sorted")
    order = np.argsort(s, kind="stable")
    s, p = s[order], p[order]
    if s.size == 1:
        return np.where(np.isclose(x, s[0]), p[0], 0.0), (s[0], s[0])
    return np.interp(x, s, p, left=0.0, right=0.0), (s[0], s[-1])


def add_profile(road, u_section, u_profile, v_section, v_profile, operation="add"):
    """Add ``fu(u) * fv(v)`` to the grid (see :meth:`pyopencrg.CrgRoad.add_profile`).

    With ``operation="multiply"`` the grid is multiplied by ``fu(u) * fv(v)``
    inside the area covered by both sections.
    """
    fu, ur = _profile_values(road.u, u_section, u_profile, "u")
    fv, vr = _profile_values(road.v, v_section, v_profile, "v")
    z = road.z.copy()
    prof = np.outer(fu, fv)
    if operation == "add":
        z += prof
    elif operation == "multiply":
        m = np.outer((road.u >= ur[0]) & (road.u <= ur[1]), (road.v >= vr[0]) & (road.v <= vr[1]))
        z[m] *= prof[m]
    else:
        raise ValueError("operation must be 'add' or 'multiply'")
    return road._derive(z=z)


def add_random(road, u_section, u_profile, v_section, v_profile, seed=0, amp_min=0.0, amp_max=0.01,
               u_smooth=0.0, v_smooth=0.0, operation="add"):
    """Add a random roughness weighted by profiles along u and v.

    Parameters
    ----------
    seed : int
        Seed of the random generator (same seed = same surface).
    amp_min, amp_max : float
        Range of the random values (m) before weighting.
    u_smooth, v_smooth : float in [0, 1]
        Smoothing along u and v, as a fraction of the road length / width.
    """
    from scipy.ndimage import uniform_filter

    for name, val in (("u_smooth", u_smooth), ("v_smooth", v_smooth)):
        if not 0 <= val <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    fu, ur = _profile_values(road.u, u_section, u_profile, "u")
    fv, vr = _profile_values(road.v, v_section, v_profile, "v")
    rng = np.random.default_rng(seed)
    nu, nv = road.shape
    zr = rng.uniform(-1.0, 1.0, (nu, nv))
    ku = min(int(round(u_smooth * nu + 1)), nu)
    kv = min(int(round(v_smooth * nv + 1)), nv)
    if ku > 1 or kv > 1:
        zr = uniform_filter(zr, size=(ku, kv), mode="constant")
    zr -= zr.min()
    mx = zr.max()
    if mx > 0:
        zr /= mx
    zr = (amp_min + (amp_max - amp_min) * zr) * np.outer(fu, fv)
    z = road.z.copy()
    if operation == "add":
        z += zr
    elif operation == "multiply":
        m = np.outer((road.u >= ur[0]) & (road.u <= ur[1]), (road.v >= vr[0]) & (road.v <= vr[1]))
        z[m] *= zr[m]
    else:
        raise ValueError("operation must be 'add' or 'multiply'")
    return road._derive(z=z)


def engrave_image(road, image_path, height=0.01, width=1.0, u0=None, v0=None, rotate=False,
                  mirror=False, threshold=127):
    """Raise the road surface where a black and white image is white.

    See :meth:`pyopencrg.CrgRoad.engrave_image`.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("engrave_image needs the Pillow package: pip install pillow") from exc
    img = Image.open(image_path).convert("L")
    if rotate:
        img = img.rotate(90, expand=True)
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w_px, h_px = img.size
    v_step = road.v_inc if road.v_inc is not None else float(np.mean(np.diff(road.v)))
    n_u = max(1, int(round(width / road.u_inc)))
    n_v = max(1, int(round(width * h_px / w_px / v_step)))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    arr = np.asarray(img.resize((n_u, n_v), resample)) > threshold  # (rows = v, cols = u)
    mask = arr[::-1, :].T  # (u, v), top of the image on the left side of the road
    i0 = road.uv_to_index(road.u_start if u0 is None else u0)
    j0 = 0 if v0 is None else int(np.clip(np.searchsorted(road.v, v0), 0, road.nv - 1))
    i1 = min(road.nu, i0 + mask.shape[0])
    j1 = min(road.nv, j0 + mask.shape[1])
    z = road.z.copy()
    z[i0:i1, j0:j1] += height * mask[: i1 - i0, : j1 - j0]
    return road._derive(z=z)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of :func:`compare`.

    Attributes
    ----------
    identical : bool
        True if both roads have the same grid and the same surface (within
        ``check_eps``).
    messages : list of str
        Differences found in the grid definition.
    u, v : numpy.ndarray
        Grid where both roads were evaluated.
    diff : numpy.ndarray
        Absolute difference of the surfaces, shape (len(u), len(v)).
    rel_diff : numpy.ndarray
        Relative difference in percent.
    """

    identical: bool
    messages: list = field(default_factory=list)
    u: np.ndarray = None
    v: np.ndarray = None
    diff: np.ndarray = None
    rel_diff: np.ndarray = None

    @property
    def max_diff(self) -> float:
        """Largest absolute difference (m)."""
        return float(np.nanmax(self.diff)) if np.isfinite(self.diff).any() else float("nan")

    @property
    def mean_diff(self) -> float:
        """Mean absolute difference (m)."""
        return float(np.nanmean(self.diff)) if np.isfinite(self.diff).any() else float("nan")

    def __str__(self):
        lines = [f"identical: {self.identical}", f"max difference: {self.max_diff:.6g} m",
                 f"mean difference: {self.mean_diff:.6g} m"]
        lines += [f"- {m}" for m in self.messages]
        return "\n".join(lines)

    def show(self, block=True):
        """Plot the differences."""
        from .plot import comparison
        return comparison(self, block=block)


def compare(a, b):
    """Compare two roads.

    The grid parameters are compared, then both surfaces are evaluated on
    the grid of ``a`` and subtracted.

    Returns
    -------
    ComparisonResult
    """
    ceps = max(a.get_option("check_eps"), b.get_option("check_eps"))
    ctol = max(a.get_option("check_tol"), b.get_option("check_tol"))
    msgs = []

    def differ(x, y):
        return abs(x - y) > max(ceps * (abs(x) + abs(y)), ctol)

    for name in ("u_start", "u_end", "u_inc", "v_min", "v_max"):
        if differ(getattr(a, name), getattr(b, name)):
            msgs.append(f"{name} differs: {getattr(a, name):.9g} / {getattr(b, name):.9g}")
    for name, i in (("x start", 0), ("x end", -1)):
        if differ(a.x[i], b.x[i]):
            msgs.append(f"{name} differs: {a.x[i]:.9g} / {b.x[i]:.9g}")
    for name, i in (("y start", 0), ("y end", -1)):
        if differ(a.y[i], b.y[i]):
            msgs.append(f"{name} differs: {a.y[i]:.9g} / {b.y[i]:.9g}")
    if a.shape != b.shape:
        msgs.append(f"grid size differs: {a.shape} / {b.shape}")
    if (a.v_inc is None) != (b.v_inc is None) or (a.v_inc is not None and differ(a.v_inc, b.v_inc)):
        msgs.append("lateral spacing differs")
    uu, vv = np.meshgrid(a.u, a.v, indexing="ij")
    za = a.uv_to_z(uu, vv)
    zb = b.uv_to_z(uu, vv)
    diff = np.abs(za - zb)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (zb / za - 1.0) * 100.0
    same_nan = np.isnan(za) == np.isnan(zb)
    ok_vals = np.all(same_nan) and np.all((diff < ceps) | np.isnan(diff))
    return ComparisonResult(identical=bool(ok_vals and not msgs), messages=msgs, u=a.u, v=a.v,
                            diff=diff, rel_diff=rel)

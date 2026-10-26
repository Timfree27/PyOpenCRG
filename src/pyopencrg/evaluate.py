"""
Evaluation of a road: elevation, gradient, normal, position, heading and curvature.

OpenCRG uses two coordinate systems:

* the **road coordinates** (u, v): ``u`` is the distance along the reference
  line, ``v`` the lateral distance to it (positive to the left);
* the **inertial coordinates** (x, y, z) of the "world".

All functions of this module are vectorised: they take numpy arrays of any
size and process them at once (in chunks, to keep the memory low). They are
used by the methods of :class:`pyopencrg.CrgRoad` (``road.uv_to_z(...)``,
``road.xy_to_uv(...)``...) and by :class:`pyopencrg.RoadEvaluator`, which are
the recommended way to call them.

The algorithms follow the ones of the OpenCRG reference implementation:

* bilinear interpolation of the elevation grid, with the same special
  handling of NaN values (a NaN corner only matters if the point is not
  exactly on the opposite edge);
* border modes (NaN, zero, keep last, repeat, reflect) and smoothing zones;
* x/y positions along the curved reference line obtained by interpolating
  between "corner normals", which gives a continuous mapping;
* inverse mapping x/y -> u/v by searching the right interval of the
  reference line, starting from the closest reference line point or from a
  hint given by the user.

The gradient and the normal are the exact derivatives of this interpolation
(including the reference line elevation, the banking and the curvature of
the u/v -> x/y mapping).
"""

from __future__ import annotations

import numpy as np

_CHUNK = 1 << 20


def _clip(a, lo, hi):
    """Faster than np.clip for small arrays."""
    return np.minimum(np.maximum(a, lo), hi)


def _chunked(func, n, *arrays):
    """Call ``func`` on slices of ``arrays`` of at most _CHUNK elements."""
    if n <= _CHUNK:
        return func(*arrays)
    outs = None
    for s in range(0, n, _CHUNK):
        res = func(*(a[s:s + _CHUNK] for a in arrays))
        if not isinstance(res, tuple):
            res = (res,)
        if outs is None:
            outs = [[] for _ in res]
        for o, r in zip(outs, res):
            o.append(r)
    outs = tuple(np.concatenate(o) for o in outs)
    return outs if len(outs) > 1 else outs[0]


# ---------------------------------------------------------------------------
# Closed track (option refline_continuation = 1)
# ---------------------------------------------------------------------------

def closure(road):
    """Information needed to close the track.

    Returns (u_start_crossing, u_end_crossing, closed_length). The closed
    length is 0 when the extrapolated ends of the reference line cannot be
    connected (then the track cannot be closed).
    """
    cache = road._cache
    if "closure" in cache:
        return cache["closure"]
    ubeg, uend = road.u_start, road.u_end
    res = (ubeg, uend, 0.0)
    if not road.is_straight:
        xbeg, ybeg, xend, yend = road.x[0], road.y[0], road.x[-1], road.y[-1]
        pbec, pbes = np.cos(road.phi_start), np.sin(road.phi_start)
        penc, pens = np.cos(road.phi_end), np.sin(road.phi_end)
        dot = pbec * penc + pbes * pens
        tol = road.get_option("check_tol")
        if dot >= 0.5:
            duend = (pbes * (ybeg - yend) + pbec * (xbeg - xend)) / dot
            dubeg = (pens * (yend - ybeg) + penc * (xend - xbeg)) / dot
            if duend >= -tol and dubeg <= tol:
                ubex = ubeg + dubeg * dot / 2
                uenx = uend + duend * dot / 2
                res = (ubex, uenx, uenx - ubex)
                cross = pbec * pens - pbes * penc
                if abs(cross) >= np.finfo(float).eps:
                    duend = (pbec * (ybeg - yend) - pbes * (xbeg - xend)) / cross
                    dubeg = (penc * (ybeg - yend) - pens * (xbeg - xend)) / cross
                    if duend >= -tol and dubeg <= tol:
                        res = (ubeg + dubeg, uend + duend, uend + duend - ubeg - dubeg)
    cache["closure"] = res
    return res


# ---------------------------------------------------------------------------
# Evaluation parameters, prepared once per road
# ---------------------------------------------------------------------------

class Params:
    """Everything the evaluation needs, read once from a road.

    Building it reads the options and the arrays of the road; evaluating
    many small sets of points then costs almost nothing more than the
    computation itself.
    """

    def __init__(self, road):
        opt = road.get_option
        self.cache = road._cache  # geometric caches (corner normals, search tree)
        self.ubeg, self.uinc, self.nu = road.u_start, road.u_inc, road.nu
        self.uend = road.u_end
        self.z = road.z
        self.v = road.v
        self.nv = self.v.size
        self.vmin, self.vmax = float(self.v[0]), float(self.v[-1])
        self.vinc = road.v_inc
        self.x, self.y, self.zr, self.b = road.x, road.y, road.zr, road.banking
        self.straight = road.is_straight
        self.phi_start, self.phi_end = road.phi_start, road.phi_end
        self.pbc, self.pbs = np.cos(self.phi_start), np.sin(self.phi_start)
        self.pec, self.pes = np.cos(self.phi_end), np.sin(self.phi_end)
        self.bdmu, self.bdmv = int(opt("border_mode_u")), int(opt("border_mode_v"))
        self.bdou, self.bdov = float(opt("border_offset_u")), float(opt("border_offset_v"))
        self.bdss, self.bdse = float(opt("border_smooth_ubeg")), float(opt("border_smooth_uend"))
        self.ceps = float(opt("check_eps"))
        self.closed = False
        self.ubex = self.uenx = self.ulex = 0.0
        if int(opt("refline_continuation")) == 1:
            self.ubex, self.uenx, self.ulex = closure(road)
            self.closed = self.ulex != 0
        self._road = road

    def map_closed(self, pu):
        if self.closed:
            return np.mod(pu - self.ubex, self.ulex) + self.ubex
        return pu


def params(road) -> Params:
    """Evaluation parameters of a road (cached, rebuilt when the road changes)."""
    key = (id(road._z), id(road._v), id(road._zr), id(road._banking), tuple(sorted(road.opts.items())))
    cached = road._cache.get("params")
    if cached is not None and cached[0] == key:
        return cached[1]
    p = Params(road)
    road._cache["params"] = (key, p)
    return p


# ---------------------------------------------------------------------------
# u, v -> z (and its gradient)
# ---------------------------------------------------------------------------

def uv_to_z(road, pu, pv, grid_only: bool = False):
    """Elevation z at road coordinates (u, v) (1D arrays of the same size).

    With ``grid_only=True`` only the elevation grid is interpolated (without
    the reference line elevation, slope, banking, border offsets and
    smoothing zones).
    """
    p = params(road)
    pu = np.asarray(pu, dtype=np.float64).ravel()
    pv = np.asarray(pv, dtype=np.float64).ravel()
    return _chunked(lambda a, b: _uv_to_z(p, a, b, grid_only, False), pu.size, pu, pv)


def uv_to_z_gradient(road, pu, pv, grid_only: bool = False):
    """Elevation z and its derivatives (dz/du, dz/dv) at road coordinates (u, v)."""
    p = params(road)
    pu = np.asarray(pu, dtype=np.float64).ravel()
    pv = np.asarray(pv, dtype=np.float64).ravel()
    return _chunked(lambda a, b: _uv_to_z(p, a, b, grid_only, True), pu.size, pu, pv)


def _wrap(t, length, reflect):
    """Repeat or reflect mapping of t (>= 0 after shift) into [0, length]; returns (t, sign of dt)."""
    if reflect:
        w = np.mod(t, 2 * length)
        back = w > length
        return np.where(back, 2 * length - w, w), np.where(back, -1.0, 1.0)
    return np.mod(t, length), None


def _uv_to_z(p, pu, pv, grid_only, grad):
    z = p.z
    nu, nv = p.nu, p.nv
    ubeg, uinc, uend = p.ubeg, p.uinc, p.uend
    vmin, vmax = p.vmin, p.vmax
    su = sv = None  # sign of the derivative after a reflection

    pu = p.map_closed(pu)
    if p.bdmu in (3, 4):
        t, su = _wrap(pu - ubeg, uend - ubeg, p.bdmu == 4)
        pu = t + ubeg
    if p.bdmv in (3, 4):
        t, sv = _wrap(pv - vmin, vmax - vmin, p.bdmv == 4)
        pv = t + vmin

    ui = (pu - ubeg) / uinc
    iu = _clip(np.floor(ui), 0, nu - 2).astype(np.int64)
    du = ui - iu
    tu = _clip(du, 0.0, 1.0)
    if p.vinc is not None:
        vi = (pv - vmin) / p.vinc
        iv = _clip(np.floor(vi), 0, nv - 2).astype(np.int64)
        dv = vi - iv
        vstep = p.vinc
    else:
        v = p.v
        iv = np.clip(np.searchsorted(v, pv, side="right") - 1, 0, nv - 2)
        vstep = v[iv + 1] - v[iv]
        dv = (pv - v[iv]) / vstep
    tv = _clip(dv, 0.0, 1.0)

    z00 = z[iu, iv]
    z10 = z[iu + 1, iv]
    z01 = z[iu, iv + 1]
    z11 = z[iu + 1, iv + 1]
    a10 = z10 - z00
    a01 = z01 - z00
    d11 = z11 - z10 - z01 + z00
    res = (d11 * tv + a10) * tu + a01 * tv + z00
    if grad:
        gu = (d11 * tv + a10) / uinc
        gv = (d11 * tu + a01) / vstep
    nanc = np.isnan(d11)
    if nanc.any():
        # a NaN corner is only relevant if the point is not on the opposite edge
        ceps = p.ceps
        uw = tu[nanc].copy()
        vw = tv[nanc].copy()
        uw[uw < ceps] = 0.0
        uw[uw > 1 - ceps] = 1.0
        vw[vw < ceps] = 0.0
        vw[vw > 1 - ceps] = 1.0
        b00, b10, b01, b11 = z00[nanc], z10[nanc], z01[nanc], z11[nanc]
        n00, n10, n01, n11 = np.isnan(b00), np.isnan(b10), np.isnan(b01), np.isnan(b11)
        relevant = ((n00 & (uw < 1) & (vw < 1)) | (n10 & (uw > 0) & (vw < 1))
                    | (n01 & (uw < 1) & (vw > 0)) | (n11 & (uw > 0) & (vw > 0)))
        b00, b10, b01, b11 = (np.where(n, 0.0, a) for n, a in ((n00, b00), (n10, b10), (n01, b01), (n11, b11)))
        e11 = b11 - b10 - b01 + b00
        val = (e11 * vw + (b10 - b00)) * uw + (b01 - b00) * vw + b00
        val[relevant] = np.nan
        res[nanc] = val
        if grad:
            g1 = (e11 * vw + (b10 - b00)) / uinc
            g2 = (e11 * uw + (b01 - b00)) / (vstep if np.isscalar(vstep) else vstep[nanc])
            g1[relevant] = np.nan
            g2[relevant] = np.nan
            gu[nanc] = g1
            gv[nanc] = g2

    out_u = tu != du
    out_v = tv != dv
    if grad:  # beyond the grid the value is kept constant
        gu[out_u] = 0.0
        gv[out_v] = 0.0

    if not grid_only:
        zr, b = p.zr, p.b
        dzr = zr[iu + 1] - zr[iu]
        db = b[iu + 1] - b[iu]
        vc = _clip(pv, vmin, vmax)
        bb = b[iu] + tu * db
        res += zr[iu] + tu * dzr + vc * bb
        if grad:
            gu += np.where(out_u, 0.0, (dzr + vc * db) / uinc)
            gv += np.where(out_v, 0.0, bb)
        if p.bdmu == 1:
            res[out_u] = p.bdou
            if grad:
                gu[out_u] = 0.0
                gv[out_u] = 0.0
        elif p.bdmu == 2 and p.bdou != 0:
            res[out_u] += p.bdou
        if p.bdmv == 1:
            res[out_v] = p.bdov
            if grad:
                gu[out_v] = 0.0
                gv[out_v] = 0.0
        elif p.bdmv == 2 and p.bdov != 0:
            res[out_v] += p.bdov
        if p.bdss > 0:  # smoothing zone at the start: blend towards the start elevation
            urel = np.maximum(0.0, pu - ubeg)
            m = p.bdss > urel
            w = urel[m] / p.bdss
            if grad:
                gu[m] = w * gu[m] + (res[m] - zr[0]) / p.bdss * (pu[m] > ubeg)
                gv[m] = w * gv[m]
            res[m] = w * (res[m] - zr[0]) + zr[0]
        if p.bdse > 0:  # smoothing zone at the end
            urel = np.maximum(0.0, uend - pu)
            m = p.bdse > urel
            w = urel[m] / p.bdse
            if grad:
                gu[m] = w * gu[m] - (res[m] - zr[-1]) / p.bdse * (pu[m] < uend)
                gv[m] = w * gv[m]
            res[m] = w * (res[m] - zr[-1]) + zr[-1]

    if p.bdmu == 0:
        res[out_u] = np.nan
    if p.bdmv == 0:
        res[out_v] = np.nan
    if not grad:
        return res
    if su is not None:
        gu *= su
    if sv is not None:
        gv *= sv
    if p.bdmu == 0:
        gu[out_u] = gv[out_u] = np.nan
    if p.bdmv == 0:
        gu[out_v] = gv[out_v] = np.nan
    return res, gu, gv


# ---------------------------------------------------------------------------
# u, v -> x, y (and the jacobian of this mapping)
# ---------------------------------------------------------------------------

def _corner_normals(p):
    """Scaled normals at the start (n1) and the end (n2) of each interval."""
    cache = p.cache
    if "normals" in cache:
        return cache["normals"]
    x, y = p.x, p.y
    nu = x.size
    dx, dy = np.diff(x), np.diff(y)
    ln = np.hypot(dx, dy)
    n12 = np.column_stack((-dy / ln, dx / ln))
    nvert = np.zeros((nu, 2))
    if nu > 2:
        tx, ty = x[2:] - x[:-2], y[2:] - y[:-2]
        lt = np.hypot(tx, ty)
        nvert[1:-1, 0] = -ty / lt
        nvert[1:-1, 1] = tx / lt
    n1 = n12.copy()
    n2 = n12.copy()
    if nu > 2:
        a = nvert[1:-1]  # vertices 1 .. nu-2
        # start normal of intervals 1 .. nu-2
        dot1 = np.sum(a * n12[1:], axis=1)
        n1[1:] = a / dot1[:, None]
        # end normal of intervals 0 .. nu-3
        dot2 = np.sum(a * n12[:-1], axis=1)
        n2[:-1] = a / dot2[:, None]
    cache["normals"] = (np.ascontiguousarray(n1), np.ascontiguousarray(n2))
    return cache["normals"]


def uv_to_xy(road, pu, pv):
    """Inertial position (x, y) of road coordinates (u, v)."""
    p = params(road)
    pu = np.asarray(pu, dtype=np.float64).ravel()
    pv = np.asarray(pv, dtype=np.float64).ravel()
    return _chunked(lambda a, b: _uv_to_xy(p, a, b, False), pu.size, pu, pv)


def _uv_to_xy(p, pu, pv, jac):
    """Positions; with ``jac=True`` also the derivatives dx/du, dy/du, dx/dv, dy/dv."""
    ubeg, uinc = p.ubeg, p.uinc
    if p.straight:
        c, s = p.pbc, p.pbs
        du = pu - ubeg
        px, py = p.x[0] + c * du - s * pv, p.y[0] + s * du + c * pv
        if not jac:
            return px, py
        one = np.ones_like(pu)
        return px, py, c * one, s * one, -s * one, c * one
    pu = p.map_closed(pu)
    x, y = p.x, p.y
    nu = x.size
    ui = (pu - ubeg) / uinc
    iu = _clip(np.floor(ui), 0, nu - 2).astype(np.int64)
    t = ui - iu
    n1, n2 = _corner_normals(p)
    n1x, n1y = n1[iu, 0], n1[iu, 1]
    n2x, n2y = n2[iu, 0], n2[iu, 1]
    xa = x[iu] + pv * n1x
    ya = y[iu] + pv * n1y
    xb = x[iu + 1] + pv * n2x
    yb = y[iu + 1] + pv * n2y
    px = (xb - xa) * t + xa
    py = (yb - ya) * t + ya
    if jac:
        xu, yu = (xb - xa) / uinc, (yb - ya) / uinc
        xv, yv = n1x + t * (n2x - n1x), n1y + t * (n2y - n1y)
    before = t < 0
    if before.any():
        c, s = p.pbc, p.pbs
        du = t[before] * uinc
        px[before] = x[0] + du * c - pv[before] * s
        py[before] = y[0] + du * s + pv[before] * c
        if jac:
            xu[before], yu[before], xv[before], yv[before] = c, s, -s, c
    after = t > 1
    if after.any():
        c, s = p.pec, p.pes
        du = (t[after] - 1) * uinc
        px[after] = x[-1] + du * c - pv[after] * s
        py[after] = y[-1] + du * s + pv[after] * c
        if jac:
            xu[after], yu[after], xv[after], yv[after] = c, s, -s, c
    if not jac:
        return px, py
    return px, py, xu, yu, xv, yv


# ---------------------------------------------------------------------------
# normal vector of the surface
# ---------------------------------------------------------------------------

def uv_to_normal(road, pu, pv):
    """Elevation and unit normal (nx, ny, nz) of the road surface, in the x/y/z frame."""
    p = params(road)
    pu = np.asarray(pu, dtype=np.float64).ravel()
    pv = np.asarray(pv, dtype=np.float64).ravel()
    return _chunked(lambda a, b: _uv_to_normal(p, a, b), pu.size, pu, pv)


def _uv_to_normal(p, pu, pv):
    z, gu, gv = _uv_to_z(p, pu, pv, False, True)
    _, _, xu, yu, xv, yv = _uv_to_xy(p, pu, pv, True)
    # chain rule: dz/du = zx xu + zy yu, dz/dv = zx xv + zy yv
    det = xu * yv - yu * xv
    zx = (gu * yv - yu * gv) / det
    zy = (xu * gv - gu * xv) / det
    inv = 1.0 / np.sqrt(1.0 + zx * zx + zy * zy)
    return z, -zx * inv, -zy * inv, inv


# ---------------------------------------------------------------------------
# x, y -> u, v
# ---------------------------------------------------------------------------

def _nearest_points(p, px, py):
    """Index of the closest reference line point of every (x, y) point."""
    cache = p.cache
    try:
        if "kdtree" not in cache:
            from scipy.spatial import cKDTree
            cache["kdtree"] = cKDTree(np.column_stack((p.x, p.y)))
        return cache["kdtree"].query(np.column_stack((px, py)))[1]
    except ImportError:
        pass
    # numpy only: closest point of a coarse sample, then of its neighbourhood
    x, y = p.x, p.y
    nu = x.size
    step = max(1, nu // 2000)
    coarse = np.arange(0, nu, step)
    out = np.empty(px.size, dtype=np.int64)
    win = np.arange(-step, step + 1)
    for s in range(0, px.size, 2048):
        qx, qy = px[s:s + 2048, None], py[s:s + 2048, None]
        k0 = coarse[np.argmin((qx - x[coarse]) ** 2 + (qy - y[coarse]) ** 2, axis=1)]
        cand = np.clip(k0[:, None] + win[None, :], 0, nu - 1)
        k = np.argmin((qx - x[cand]) ** 2 + (qy - y[cand]) ** 2, axis=1)
        out[s:s + 2048] = cand[np.arange(cand.shape[0]), k]
    return out


def xy_to_uv(road, px, py, u_hint=None):
    """Road coordinates (u, v) of inertial positions (x, y).

    Parameters
    ----------
    u_hint : array_like, optional
        Approximate u of the points (one value for all of them, or one per
        point). The search then starts there instead of at the closest
        reference line point: faster, and needed when the road passes
        several times close to the same place (crossings, overpasses).
    """
    p = params(road)
    px = np.asarray(px, dtype=np.float64).ravel()
    py = np.asarray(py, dtype=np.float64).ravel()
    if u_hint is None:
        return _chunked(lambda a, b: _xy_to_uv(p, a, b, None), px.size, px, py)
    uh = np.broadcast_to(np.asarray(u_hint, dtype=np.float64), px.shape).ravel()
    return _chunked(lambda a, b, c: _xy_to_uv(p, a, b, c), px.size, px, py, uh)


def _xy_to_uv(p, px, py, u_hint):
    ubeg, uinc = p.ubeg, p.uinc
    x0, y0 = p.x[0], p.y[0]
    pbc, pbs = p.pbc, p.pbs
    if p.straight:
        dx, dy = px - x0, py - y0
        return ubeg + pbc * dx + pbs * dy, -pbs * dx + pbc * dy
    x, y = p.x, p.y
    nu = x.size
    if u_hint is None:
        j = _clip(_nearest_points(p, px, py) - 1, 0, nu - 2)
    else:
        j = _clip(np.floor((u_hint - ubeg) / uinc), 0, nu - 2).astype(np.int64)
        # refine the hint of every point by projection on the local tangent
        s = ((px - x[j]) * (x[j + 1] - x[j]) + (py - y[j]) * (y[j + 1] - y[j])) / (uinc * uinc)
        j = _clip(j + np.floor(s).astype(np.int64), 0, nu - 2)

    def corners(jj):
        jm, j3 = np.maximum(jj - 1, 0), np.minimum(jj + 2, nu - 1)
        return x[jm], y[jm], x[jj], y[jj], x[jj + 1], y[jj + 1], x[j3], y[j3]

    # walk interval by interval until the point lies between the normals of both ends
    first = True
    active = None
    while True:
        if first:
            x0_, y0_, x1, y1, x2, y2, x3, y3 = corners(j)
            hdf = (px - x2) * (x3 - x1) + (py - y2) * (y3 - y1)
            hdb = (px - x1) * (x2 - x0_) + (py - y1) * (y2 - y0_)
            jj, f, b = j, hdf, hdb
        else:
            jj = j[active]
            qx, qy = px[active], py[active]
            c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y = corners(jj)
            f = (qx - c2x) * (c3x - c1x) + (qy - c2y) * (c3y - c1y)
            b = (qx - c1x) * (c2x - c0x) + (qy - c1y) * (c2y - c0y)
        fwd = (f > 0) & (jj < nu - 2)
        bwd = (b < 0) & (jj > 0) & ~fwd
        move = fwd | bwd
        if not move.any():
            break
        idx = np.flatnonzero(move) if first else active[move]
        j[idx] += np.where(fwd[move], 1, -1)
        active = idx
        first = False
    if not first:  # some points moved: final values for every point
        x0_, y0_, x1, y1, x2, y2, x3, y3 = corners(j)
        hdf = (px - x2) * (x3 - x1) + (py - y2) * (y3 - y1)
        hdb = (px - x1) * (x2 - x0_) + (py - y1) * (y2 - y0_)
    x2x1, y2y1 = x2 - x1, y2 - y1
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (x2x1 * (py - y1) - y2y1 * (px - x1)) / np.hypot(x2x1, y2y1)
        ta = hdb / ((x2 - x0_) * x2x1 + (y2 - y0_) * y2y1)
        tb = ((x3 - x1) * (x2 - px) + (y3 - y1) * (y2 - py)) / ((x3 - x1) * x2x1 + (y3 - y1) * y2y1)
        u = ubeg + (j + ta / (ta + tb)) * uinc

    uend = p.uend
    xe, ye = x[-1], y[-1]
    pec, pes = p.pec, p.pes
    before = hdb < 0
    if before.any():
        dx, dy = px[before] - x0, py[before] - y0
        ub = ubeg + pbc * dx + pbs * dy
        vb = -pbs * dx + pbc * dy
        if p.closed:
            sw = (ub < p.ubex) | ((ub > uend) & (ub < p.uenx))
            dx2, dy2 = px[before][sw] - xe, py[before][sw] - ye
            ub[sw] = uend + pec * dx2 + pes * dy2
            vb[sw] = -pes * dx2 + pec * dy2
        u[before], v[before] = ub, vb
    after = (hdf > 0) & ~before
    if after.any():
        dx, dy = px[after] - xe, py[after] - ye
        ua = uend + pec * dx + pes * dy
        va = -pes * dx + pec * dy
        if p.closed:
            sw = ua > p.uenx
            dx2, dy2 = px[after][sw] - x0, py[after][sw] - y0
            ua[sw] = ubeg + pbc * dx2 + pbs * dy2
            va[sw] = -pbs * dx2 + pbc * dy2
        u[after], v[after] = ua, va
    return u, v


# ---------------------------------------------------------------------------
# u -> heading, curvature ; u, v -> grid indices
# ---------------------------------------------------------------------------

def u_to_phi(road, pu):
    """Heading angle (rad) of the reference line at u."""
    p = params(road)
    pu = np.asarray(pu, dtype=np.float64).ravel()
    if p.straight:
        return np.full(pu.shape, p.phi_start)
    pu = p.map_closed(pu)
    phi = road.phi
    iu = np.floor((pu - p.ubeg) / p.uinc).astype(np.int64)
    out = phi[np.clip(iu, 0, phi.size - 1)]
    out = np.where(iu < 0, p.phi_start, out)
    return np.where(iu > phi.size - 1, p.phi_end, out)


def u_to_curvature(road, pu):
    """Curvature (1/m, positive when turning left) of the reference line at u."""
    p = params(road)
    pu = np.asarray(pu, dtype=np.float64).ravel()
    rc = road.curvature
    if p.straight or rc.size == 0:
        return np.zeros(pu.shape)
    pu = p.map_closed(pu)
    iu = np.floor((pu - p.ubeg) / p.uinc).astype(np.int64)
    ok = (iu >= 1) & (iu <= rc.size)
    return np.where(ok, rc[np.clip(iu - 1, 0, rc.size - 1)], 0.0)


def uv_to_index(road, pu, pv=None):
    """Grid indices (0 based) of the cells containing (u, v), clipped to the grid."""
    p = params(road)
    pu = p.map_closed(np.asarray(pu, dtype=np.float64).ravel())
    iu = np.minimum(np.floor(np.maximum((pu - p.ubeg) / p.uinc, 0)), p.nu - 1).astype(np.int64)
    if pv is None:
        return iu
    pv = np.asarray(pv, dtype=np.float64).ravel()
    v = p.v
    if p.vinc is not None:
        iv = np.minimum(np.floor(np.maximum((pv - v[0]) / p.vinc, 0)), p.nv - 1).astype(np.int64)
    else:
        iv = np.clip(np.searchsorted(v, np.clip(pv, v[0], v[-1]), side="right") - 1, 0, p.nv - 1)
    return iu, iv

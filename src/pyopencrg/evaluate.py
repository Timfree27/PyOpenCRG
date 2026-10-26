"""
Evaluation of a road: elevation, position, heading and curvature.

OpenCRG uses two coordinate systems:

* the **road coordinates** (u, v): ``u`` is the distance along the reference
  line, ``v`` the lateral distance to it (positive to the left);
* the **inertial coordinates** (x, y, z) of the "world".

All functions of this module are vectorised: they take numpy arrays of any
size and process them at once (in chunks, to keep the memory low). They are
used by the methods of :class:`pyopencrg.CrgRoad` (``road.uv_to_z(...)``,
``road.xy_to_uv(...)``...), which are the recommended way to call them.

The algorithms follow the ones of the OpenCRG reference implementation:

* bilinear interpolation of the elevation grid, with the same special
  handling of NaN values (a NaN corner only matters if the point is not
  exactly on the opposite edge);
* border modes (NaN, zero, keep last, repeat, reflect) and smoothing zones;
* x/y positions along the curved reference line obtained by interpolating
  between "corner normals", which gives a continuous mapping;
* inverse mapping x/y -> u/v by searching the right interval of the
  reference line (a KD-tree gives the starting point).
"""

from __future__ import annotations

import numpy as np

_CHUNK = 1 << 20


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


def _map_closed(road, pu):
    if int(road.get_option("refline_continuation")) == 1:
        ubex, _, ulex = closure(road)
        if ulex != 0:
            return np.mod(pu - ubex, ulex) + ubex
    return pu


# ---------------------------------------------------------------------------
# u, v -> z
# ---------------------------------------------------------------------------

def uv_to_z(road, pu, pv, grid_only: bool = False):
    """Elevation z at road coordinates (u, v).

    Parameters
    ----------
    road : CrgRoad
    pu, pv : 1D numpy arrays of the same size
    grid_only : bool
        If True, only the elevation grid is interpolated (without the
        reference line elevation, slope, banking, border offsets and
        smoothing zones).
    """
    pu = np.asarray(pu, dtype=np.float64).ravel()
    pv = np.asarray(pv, dtype=np.float64).ravel()
    return _chunked(lambda a, b: _uv_to_z(road, a, b, grid_only), pu.size, pu, pv)


def _uv_to_z(road, pu, pv, grid_only):
    opt = road.get_option
    bdmu, bdmv = int(opt("border_mode_u")), int(opt("border_mode_v"))
    ceps = opt("check_eps")
    z = road.z
    nu, nv = z.shape
    ubeg, uinc = road.u_start, road.u_inc
    uend = road.u_end
    v = road.v
    vmin, vmax = v[0], v[-1]

    pu = _map_closed(road, pu)
    if bdmu == 3:
        pu = np.mod(pu - ubeg, uend - ubeg) + ubeg
    elif bdmu == 4:
        w = np.mod(pu - ubeg, 2 * (uend - ubeg))
        pu = np.where(w > uend - ubeg, 2 * (uend - ubeg) - w, w) + ubeg
    if bdmv == 3:
        pv = np.mod(pv - vmin, vmax - vmin) + vmin
    elif bdmv == 4:
        w = np.mod(pv - vmin, 2 * (vmax - vmin))
        pv = np.where(w > vmax - vmin, 2 * (vmax - vmin) - w, w) + vmin

    ui = (pu - ubeg) / uinc
    iu = np.clip(np.floor(ui), 0, nu - 2).astype(np.int64)
    du = ui - iu
    tu = np.clip(du, 0.0, 1.0)
    vinc = road.v_inc
    if vinc is not None:
        vi = (pv - vmin) / vinc
        iv = np.clip(np.floor(vi), 0, nv - 2).astype(np.int64)
        dv = vi - iv
    else:
        iv = np.clip(np.searchsorted(v, pv, side="right") - 1, 0, nv - 2)
        dv = (pv - v[iv]) / (v[iv + 1] - v[iv])
    tv = np.clip(dv, 0.0, 1.0)

    z00 = z[iu, iv]
    z10 = z[iu + 1, iv]
    z01 = z[iu, iv + 1]
    z11 = z[iu + 1, iv + 1]
    d11 = z11 - z10 - z01 + z00
    res = (d11 * tv + (z10 - z00)) * tu + (z01 - z00) * tv + z00
    nanc = np.isnan(d11)
    if nanc.any():
        # a NaN corner is only relevant if the point is not on the opposite edge
        uw = tu[nanc].copy()
        vw = tv[nanc].copy()
        uw[uw < ceps] = 0.0
        uw[uw > 1 - ceps] = 1.0
        vw[vw < ceps] = 0.0
        vw[vw > 1 - ceps] = 1.0
        a00, a10, a01, a11 = z00[nanc], z10[nanc], z01[nanc], z11[nanc]
        n00, n10, n01, n11 = np.isnan(a00), np.isnan(a10), np.isnan(a01), np.isnan(a11)
        relevant = ((n00 & (uw < 1) & (vw < 1)) | (n10 & (uw > 0) & (vw < 1))
                    | (n01 & (uw < 1) & (vw > 0)) | (n11 & (uw > 0) & (vw > 0)))
        a00, a10, a01, a11 = (np.where(n, 0.0, a) for n, a in ((n00, a00), (n10, a10), (n01, a01), (n11, a11)))
        val = ((a11 - a10 - a01 + a00) * vw + (a10 - a00)) * uw + (a01 - a00) * vw + a00
        val[relevant] = np.nan
        res[nanc] = val

    out_u = tu != du
    out_v = tv != dv
    if grid_only:
        if bdmu == 0:
            res[out_u] = np.nan
        if bdmv == 0:
            res[out_v] = np.nan
        return res

    zr = road.zr
    b = road.banking
    res += zr[iu] + tu * (zr[iu + 1] - zr[iu])
    res += np.clip(pv, vmin, vmax) * (b[iu] + tu * (b[iu + 1] - b[iu]))
    bdou, bdov = opt("border_offset_u"), opt("border_offset_v")
    if bdmu == 1:
        res[out_u] = bdou
    elif bdmu == 2 and bdou != 0:
        res[out_u] += bdou
    if bdmv == 1:
        res[out_v] = bdov
    elif bdmv == 2 and bdov != 0:
        res[out_v] += bdov
    bdss, bdse = opt("border_smooth_ubeg"), opt("border_smooth_uend")
    if bdss > 0:
        urel = np.maximum(0.0, pu - ubeg)
        m = bdss > urel
        res[m] = urel[m] / bdss * (res[m] - zr[0]) + zr[0]
    if bdse > 0:
        urel = np.maximum(0.0, uend - pu)
        m = bdse > urel
        res[m] = urel[m] / bdse * (res[m] - zr[-1]) + zr[-1]
    if bdmu == 0:
        res[out_u] = np.nan
    if bdmv == 0:
        res[out_v] = np.nan
    return res


# ---------------------------------------------------------------------------
# u, v -> x, y
# ---------------------------------------------------------------------------

def _corner_normals(road):
    """Scaled normals at the start (n1) and the end (n2) of each interval."""
    cache = road._cache
    if "normals" in cache:
        return cache["normals"]
    x, y = road.x, road.y
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
    cache["normals"] = (n1, n2)
    return n1, n2


def uv_to_xy(road, pu, pv):
    """Inertial position (x, y) of road coordinates (u, v)."""
    pu = np.asarray(pu, dtype=np.float64).ravel()
    pv = np.asarray(pv, dtype=np.float64).ravel()
    return _chunked(lambda a, b: _uv_to_xy(road, a, b), pu.size, pu, pv)


def _uv_to_xy(road, pu, pv):
    ubeg, uinc = road.u_start, road.u_inc
    if road.is_straight:
        c, s = np.cos(road.phi_start), np.sin(road.phi_start)
        du = pu - ubeg
        return road.x[0] + c * du - s * pv, road.y[0] + s * du + c * pv
    pu = _map_closed(road, pu)
    x, y = road.x, road.y
    nu = x.size
    ui = (pu - ubeg) / uinc
    iu = np.clip(np.floor(ui), 0, nu - 2).astype(np.int64)
    t = ui - iu
    n1, n2 = _corner_normals(road)
    xa = x[iu] + pv * n1[iu, 0]
    ya = y[iu] + pv * n1[iu, 1]
    xb = x[iu + 1] + pv * n2[iu, 0]
    yb = y[iu + 1] + pv * n2[iu, 1]
    px = (xb - xa) * t + xa
    py = (yb - ya) * t + ya
    before = t < 0
    if before.any():
        c, s = np.cos(road.phi_start), np.sin(road.phi_start)
        du = t[before] * uinc
        px[before] = x[0] + du * c - pv[before] * s
        py[before] = y[0] + du * s + pv[before] * c
    after = t > 1
    if after.any():
        c, s = np.cos(road.phi_end), np.sin(road.phi_end)
        du = (t[after] - 1) * uinc
        px[after] = x[-1] + du * c - pv[after] * s
        py[after] = y[-1] + du * s + pv[after] * c
    return px, py


# ---------------------------------------------------------------------------
# x, y -> u, v
# ---------------------------------------------------------------------------

def _kdtree(road):
    cache = road._cache
    if "kdtree" not in cache:
        from scipy.spatial import cKDTree
        cache["kdtree"] = cKDTree(np.column_stack((road.x, road.y)))
    return cache["kdtree"]


def xy_to_uv(road, px, py, u_hint=None):
    """Road coordinates (u, v) of inertial positions (x, y).

    Parameters
    ----------
    u_hint : array_like, optional
        Approximate u of each point. Useful when the road passes several
        times close to the same place (crossings, overpasses): the search
        then starts at this u instead of the closest reference line point.
    """
    px = np.asarray(px, dtype=np.float64).ravel()
    py = np.asarray(py, dtype=np.float64).ravel()
    if u_hint is None:
        return _chunked(lambda a, b: _xy_to_uv(road, a, b, None), px.size, px, py)
    uh = np.broadcast_to(np.asarray(u_hint, dtype=np.float64), px.shape).ravel()
    return _chunked(lambda a, b, c: _xy_to_uv(road, a, b, c), px.size, px, py, uh)


def _xy_to_uv(road, px, py, u_hint):
    ubeg, uinc = road.u_start, road.u_inc
    x0, y0 = road.x[0], road.y[0]
    pbc, pbs = np.cos(road.phi_start), np.sin(road.phi_start)
    if road.is_straight:
        dx, dy = px - x0, py - y0
        return ubeg + pbc * dx + pbs * dy, -pbs * dx + pbc * dy
    x, y = road.x, road.y
    nu = x.size
    if u_hint is None:
        _, k = _kdtree(road).query(np.column_stack((px, py)))
        j = np.clip(k - 1, 0, nu - 2)
    else:
        j = np.clip(np.floor((u_hint - ubeg) / uinc), 0, nu - 2).astype(np.int64)

    def tests(jj, qx, qy):
        x0_, y0_ = x[np.maximum(jj - 1, 0)], y[np.maximum(jj - 1, 0)]
        x1, y1 = x[jj], y[jj]
        x2, y2 = x[jj + 1], y[jj + 1]
        j3 = np.minimum(jj + 2, nu - 1)
        x3, y3 = x[j3], y[j3]
        hdf = (qx - x2) * (x3 - x1) + (qy - y2) * (y3 - y1)
        hdb = (qx - x1) * (x2 - x0_) + (qy - y1) * (y2 - y0_)
        return hdf, hdb

    active = np.arange(px.size)
    for _ in range(nu + 1):
        jj = j[active]
        hdf, hdb = tests(jj, px[active], py[active])
        fwd = (hdf > 0) & (jj < nu - 2)
        bwd = (hdb < 0) & (jj > 0) & ~fwd
        move = fwd | bwd
        if not move.any():
            break
        j[active[fwd]] += 1
        j[active[bwd]] -= 1
        active = active[move]

    jm1 = np.maximum(j - 1, 0)
    j3 = np.minimum(j + 2, nu - 1)
    x0_, y0_ = x[jm1], y[jm1]
    x1, y1 = x[j], y[j]
    x2, y2 = x[j + 1], y[j + 1]
    x3, y3 = x[j3], y[j3]
    hdf = (px - x2) * (x3 - x1) + (py - y2) * (y3 - y1)
    hdb = (px - x1) * (x2 - x0_) + (py - y1) * (y2 - y0_)
    x2x1, y2y1 = x2 - x1, y2 - y1
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (x2x1 * (py - y1) - y2y1 * (px - x1)) / np.hypot(x2x1, y2y1)
        ta = hdb / ((x2 - x0_) * x2x1 + (y2 - y0_) * y2y1)
        tb = ((x3 - x1) * (x2 - px) + (y3 - y1) * (y2 - py)) / ((x3 - x1) * x2x1 + (y3 - y1) * y2y1)
        u = ubeg + (j + ta / (ta + tb)) * uinc

    closed = int(road.get_option("refline_continuation")) == 1
    ubex, uenx, ulex = closure(road) if closed else (0, 0, 0)
    uend = road.u_end
    xe, ye = road.x[-1], road.y[-1]
    pec, pes = np.cos(road.phi_end), np.sin(road.phi_end)

    before = hdb < 0
    if before.any():
        dx, dy = px[before] - x0, py[before] - y0
        ub = ubeg + pbc * dx + pbs * dy
        vb = -pbs * dx + pbc * dy
        if closed and ulex != 0:
            sw = (ub < ubex) | ((ub > uend) & (ub < uenx))
            dx2, dy2 = px[before][sw] - xe, py[before][sw] - ye
            ub[sw] = uend + pec * dx2 + pes * dy2
            vb[sw] = -pes * dx2 + pec * dy2
        u[before], v[before] = ub, vb
    after = (hdf > 0) & ~before
    if after.any():
        dx, dy = px[after] - xe, py[after] - ye
        ua = uend + pec * dx + pes * dy
        va = -pes * dx + pec * dy
        if closed and ulex != 0:
            sw = ua > uenx
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
    pu = np.asarray(pu, dtype=np.float64).ravel()
    if road.is_straight:
        return np.full(pu.shape, road.phi_start)
    pu = _map_closed(road, pu)
    phi = road.phi
    iu = np.floor((pu - road.u_start) / road.u_inc).astype(np.int64)
    out = phi[np.clip(iu, 0, phi.size - 1)]
    out = np.where(iu < 0, road.phi_start, out)
    return np.where(iu > phi.size - 1, road.phi_end, out)


def u_to_curvature(road, pu):
    """Curvature (1/m, positive when turning left) of the reference line at u."""
    pu = np.asarray(pu, dtype=np.float64).ravel()
    rc = road.curvature
    if road.is_straight or rc.size == 0:
        return np.zeros(pu.shape)
    pu = _map_closed(road, pu)
    iu = np.floor((pu - road.u_start) / road.u_inc).astype(np.int64)
    ok = (iu >= 1) & (iu <= rc.size)
    return np.where(ok, rc[np.clip(iu - 1, 0, rc.size - 1)], 0.0)


def uv_to_index(road, pu, pv=None):
    """Grid indices (0 based) of the cells containing (u, v), clipped to the grid."""
    pu = _map_closed(road, np.asarray(pu, dtype=np.float64).ravel())
    iu = np.minimum(np.floor(np.maximum((pu - road.u_start) / road.u_inc, 0)), road.nu - 1).astype(np.int64)
    if pv is None:
        return iu
    pv = np.asarray(pv, dtype=np.float64).ravel()
    v = road.v
    if road.v_inc is not None:
        iv = np.minimum(np.floor(np.maximum((pv - v[0]) / road.v_inc, 0)), road.nv - 1).astype(np.int64)
    else:
        iv = np.clip(np.searchsorted(v, np.clip(pv, v[0], v[-1]), side="right") - 1, 0, road.nv - 1)
    return iu, iv

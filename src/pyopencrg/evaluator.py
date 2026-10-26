"""
Fast evaluation for repeated calls: :class:`RoadEvaluator`.

A simulation evaluates the road thousands of times per second, each time on
a few hundred points (the contact points of the tyres). The methods of
:class:`pyopencrg.CrgRoad` are fast, but each call has a small fixed cost
(reading options, checking the inputs...). An evaluator prepares everything
once and keeps it, so that small repeated calls cost almost only the
computation itself::

    ev = road.evaluator()
    for step in range(n_steps):
        x, y = contact_points(step)                       # your simulation
        z, nx, ny, nz = ev.xy_to_z(x, y, with_normal=True)

The evaluator also remembers the last u found for each point and starts the
next search from there ("warm start"): points that move continuously are
found in one or two iterations.

For a file too large for the memory, :meth:`pyopencrg.CrgFile.evaluator`
gives a :class:`FileEvaluator` with the same methods, which keeps only a
window of the road in memory and reads it again when the points move.
"""

from __future__ import annotations

import numpy as np

from . import evaluate as _ev

__all__ = ["RoadEvaluator", "FileEvaluator"]


def _inputs(a, b):
    """Two inputs as flat float64 arrays, with the output shape (``()`` for scalars)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        a, b = np.broadcast_arrays(a, b)
    return a.shape, a.ravel(), b.ravel()


def _out(arr, shape):
    if shape == ():
        return float(arr[0])
    return arr.reshape(shape)


class RoadEvaluator:
    """Evaluator of a road prepared for fast repeated calls.

    Create it with :meth:`pyopencrg.CrgRoad.evaluator`. Its methods have the
    same names, parameters and results as the ones of the road.

    Attributes
    ----------
    road : CrgRoad
        The evaluated road.
    """

    def __init__(self, road):
        self.road = road
        self._p = _ev.params(road)
        self._last_u = None
        half = max(abs(self._p.vmin), abs(self._p.vmax))
        self._v_limit = 2.0 * half + 1.0  # beyond this lateral distance a warm start is distrusted

    def reset(self):
        """Forget the last positions (the next search starts from scratch)."""
        self._last_u = None

    # --- road coordinates ------------------------------------------------

    def uv_to_z(self, u, v, grid_only: bool = False, with_normal: bool = False):
        """Elevation at (u, v), optionally with the normal. See :meth:`pyopencrg.CrgRoad.uv_to_z`."""
        shape, pu, pv = _inputs(u, v)
        if with_normal:
            if grid_only:
                raise ValueError("with_normal=True is not available with grid_only=True")
            return tuple(_out(a, shape) for a in _ev._surface(self._p, pu, pv, 2))
        if grid_only:
            z = _ev._chunked(lambda a, b: _ev._uv_to_z(self._p, a, b, True, False), pu.size, pu, pv)
        else:
            z = _ev._surface(self._p, pu, pv, 0)
        return _out(z, shape)

    def uv_to_gradient(self, u, v, grid_only: bool = False):
        """Slopes (dz/du, dz/dv) at (u, v). See :meth:`pyopencrg.CrgRoad.uv_to_gradient`."""
        shape, pu, pv = _inputs(u, v)
        if grid_only:
            _, gu, gv = _ev._chunked(lambda a, b: _ev._uv_to_z(self._p, a, b, True, True), pu.size, pu, pv)
        else:
            _, gu, gv = _ev._surface(self._p, pu, pv, 1)
        return _out(gu, shape), _out(gv, shape)

    def uv_to_normal(self, u, v):
        """Unit normal (nx, ny, nz) at (u, v). See :meth:`pyopencrg.CrgRoad.uv_to_normal`."""
        _, nx, ny, nz = self.uv_to_z(u, v, with_normal=True)
        return nx, ny, nz

    def uv_to_xy(self, u, v):
        """Inertial position (x, y) of (u, v). See :meth:`pyopencrg.CrgRoad.uv_to_xy`."""
        shape, pu, pv = _inputs(u, v)
        x, y = _ev._chunked(lambda a, b: _ev._uv_to_xy(self._p, a, b, False), pu.size, pu, pv)
        return _out(x, shape), _out(y, shape)

    # --- inertial coordinates ---------------------------------------------

    def xy_to_uv(self, x, y, u_hint=None):
        """Road coordinates (u, v) of (x, y). See :meth:`pyopencrg.CrgRoad.xy_to_uv`.

        Without ``u_hint``, the u found at the previous call is used as the
        starting point when the number of points is the same.
        """
        shape, px, py = _inputs(x, y)
        u, v = self._xy_to_uv(px, py, u_hint)
        return _out(u, shape), _out(v, shape)

    def xy_to_z(self, x, y, u_hint=None, with_normal: bool = False):
        """Elevation at (x, y), optionally with the normal. See :meth:`pyopencrg.CrgRoad.xy_to_z`."""
        shape, px, py = _inputs(x, y)
        u, v, cell = self._xy_to_uv(px, py, u_hint, True)
        if with_normal:
            return tuple(_out(a, shape) for a in _ev._surface(self._p, u, v, 2, cell))
        return _out(_ev._surface(self._p, u, v, 0, cell), shape)

    def xy_to_normal(self, x, y, u_hint=None):
        """Unit normal (nx, ny, nz) at (x, y). See :meth:`pyopencrg.CrgRoad.xy_to_normal`."""
        _, nx, ny, nz = self.xy_to_z(x, y, u_hint, with_normal=True)
        return nx, ny, nz

    def _xy_to_uv(self, px, py, u_hint, cell=False):
        """(u, v), and with ``cell=True`` the cells of the points (see ``evaluate._fast``)."""
        p = self._p
        if u_hint is not None:
            hint = np.broadcast_to(np.asarray(u_hint, dtype=np.float64), px.shape).ravel()
        elif self._last_u is not None and self._last_u.size == px.size:
            hint = self._last_u
        else:
            hint = None
        u, v, cells = _ev._search(p, px, py, hint, True)
        if hint is not None and u_hint is None:  # warm start: check that the previous positions were still close
            bad = ~(np.abs(v) <= self._v_limit)
            if bad.any():
                u[bad], v[bad] = _ev._xy_to_uv(p, px[bad], py[bad], None)
                cells = None
        self._last_u = u.copy()
        return (u, v, cells) if cell else (u, v)


class FileEvaluator:
    """Evaluator of an open file that keeps only a window of the road in memory.

    Create it with :meth:`pyopencrg.CrgFile.evaluator`. Its methods have the
    same names, parameters and results as the ones of
    :class:`RoadEvaluator`, and give the same values as an evaluator of the
    whole road.

    Attributes
    ----------
    road : CrgRoad or None
        The window of the road in memory.
    reloads : int
        Number of windows read so far.
    """

    def __init__(self, file, window=40.0, margin=10.0, apply_mods=False, dtype="float64", check=False):
        window, margin = float(window), float(margin)
        if not window > 0 or not 0 <= margin < window / 2:
            raise ValueError("window must be positive and margin less than window / 2")
        self._file = file
        self._opts = (dtype, check, bool(apply_mods))
        self._line = RoadEvaluator(file._geometry(bool(apply_mods)))  # x, y <-> u, v on the whole line
        g = self._line._p
        self._k = g.uinc / file.u_inc  # scale_length of the modifiers
        self._window, self._margin = window, margin
        self._rows = None  # first and last row of the window
        self._p = None
        self.road = None
        self.reloads = 0

    @property
    def window(self):
        """u range (m) of the window in memory, or None."""
        if self._rows is None:
            return None
        g = self._line._p
        return g.ubeg + self._rows[0] * g.uinc, g.ubeg + self._rows[1] * g.uinc

    def reset(self):
        """Forget the last positions (the next search starts from scratch)."""
        self._line.reset()

    # --- window ------------------------------------------------------------

    def _map(self, pu):
        """u of the whole road where the surface is read, and the sign of du (None: no reflection)."""
        g = self._line._p
        pu = g.map_closed(pu)
        su = None
        if g.bdmu in (3, 4):
            t, su = _ev._wrap(pu - g.ubeg, g.uend - g.ubeg, g.bdmu == 4)
            pu = t + g.ubeg
        return pu, su

    def _ensure(self, pu):
        """Read a window that covers the u values ``pu`` with the margin."""
        g = self._line._p
        ok = pu[np.isfinite(pu)]
        if ok.size == 0:
            if self._p is not None:
                return
            lo = hi = g.ubeg
        else:  # points beyond the ends need the window that touches this end
            lo = min(max(float(ok.min()), g.ubeg), g.uend)
            hi = min(max(float(ok.max()), g.ubeg), g.uend)
        last = g.nu - 1
        if self._rows is not None:
            w0, w1 = self.window
            if ((self._rows[0] == 0 or lo >= w0 + self._margin)
                    and (self._rows[1] == last or hi <= w1 - self._margin)):
                return
        length = max(self._window, hi - lo + 2 * self._margin)
        a = 0.5 * (lo + hi) - 0.5 * length
        a = min(max(a, g.ubeg), max(g.ubeg, g.uend - length))
        i0 = max(0, int(np.floor((a - g.ubeg) / g.uinc)))
        i1 = min(last, int(np.ceil((a + length - g.ubeg) / g.uinc)))
        i0 = min(i0, i1 - 1)
        dtype, check, apply_mods = self._opts
        part = self._file._read(i0, i1 + 1, 1, 1, dtype, check, apply_mods)
        # the ends of the whole road are handled here: the window only fills the grid
        opts = part.opts
        if i0 > 0:
            opts["border_smooth_ubeg"] = 0.0
        if i1 < last:
            opts["border_smooth_uend"] = 0.0
        if g.bdmu in (3, 4):
            opts["border_mode_u"] = 2
        opts["refline_continuation"] = 0
        self.road = part
        self._p = _ev.params(part)
        self._rows = (i0, i1)
        self.reloads += 1

    def _surface(self, pu, pv, grid_only, grad, normal, cell=None):
        pm, su = self._map(pu)
        self._ensure(pm)
        p = self._p
        g = self._line._p
        if not grid_only and not g.closed and pu.min() >= g.ubeg and pu.max() <= g.uend:
            # inside the road (u not mapped): direct path on the window, which has the same reference line
            if cell is not None:
                j = cell[0] - self._rows[0]
                cell = (j, cell[1], cell[2]) if j.min() >= 0 and j.max() <= p.nu - 2 else None
            res = _ev._fast(p, pu, pv, 2 if normal else int(grad), cell)
            if res is not None:
                return res
        if not (grad or normal):
            return _ev._chunked(lambda a, b: _ev._uv_to_z(p, a, b, grid_only, False), pu.size, pm, pv)
        z, gu, gv = _ev._chunked(lambda a, b: _ev._uv_to_z(p, a, b, grid_only, True), pu.size, pm, pv)
        if su is not None:
            gu = gu * su
        if grad:
            return z, gu, gv
        # derivatives of x, y on the whole line
        _, _, xu, yu, xv, yv = _ev._chunked(lambda a, b: _ev._uv_to_xy(g, a, b, True), pu.size, pu, pv)
        return (z,) + _ev._normal(gu, gv, xu, yu, xv, yv)

    # --- road coordinates ------------------------------------------------

    def uv_to_z(self, u, v, grid_only: bool = False, with_normal: bool = False):
        """Elevation at (u, v), optionally with the normal. See :meth:`pyopencrg.CrgRoad.uv_to_z`."""
        shape, pu, pv = _inputs(u, v)
        if with_normal:
            if grid_only:
                raise ValueError("with_normal=True is not available with grid_only=True")
            return tuple(_out(a, shape) for a in self._surface(pu, pv, False, False, True))
        return _out(self._surface(pu, pv, grid_only, False, False), shape)

    def uv_to_gradient(self, u, v, grid_only: bool = False):
        """Slopes (dz/du, dz/dv) at (u, v). See :meth:`pyopencrg.CrgRoad.uv_to_gradient`."""
        shape, pu, pv = _inputs(u, v)
        _, gu, gv = self._surface(pu, pv, grid_only, True, False)
        return _out(gu, shape), _out(gv, shape)

    def uv_to_normal(self, u, v):
        """Unit normal (nx, ny, nz) at (u, v). See :meth:`pyopencrg.CrgRoad.uv_to_normal`."""
        _, nx, ny, nz = self.uv_to_z(u, v, with_normal=True)
        return nx, ny, nz

    def uv_to_xy(self, u, v):
        """Inertial position (x, y) of (u, v). See :meth:`pyopencrg.CrgRoad.uv_to_xy`."""
        return self._line.uv_to_xy(u, v)

    # --- inertial coordinates ---------------------------------------------

    def xy_to_uv(self, x, y, u_hint=None):
        """Road coordinates (u, v) of (x, y). See :meth:`RoadEvaluator.xy_to_uv`."""
        return self._line.xy_to_uv(x, y, u_hint)

    def xy_to_z(self, x, y, u_hint=None, with_normal: bool = False):
        """Elevation at (x, y), optionally with the normal. See :meth:`pyopencrg.CrgRoad.xy_to_z`."""
        shape, px, py = _inputs(x, y)
        u, v, cell = self._line._xy_to_uv(px, py, u_hint, True)
        if with_normal:
            return tuple(_out(a, shape) for a in self._surface(u, v, False, False, True, cell))
        return _out(self._surface(u, v, False, False, False, cell), shape)

    def xy_to_normal(self, x, y, u_hint=None):
        """Unit normal (nx, ny, nz) at (x, y). See :meth:`pyopencrg.CrgRoad.xy_to_normal`."""
        _, nx, ny, nz = self.xy_to_z(x, y, u_hint, with_normal=True)
        return nx, ny, nz

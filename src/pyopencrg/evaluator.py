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
"""

from __future__ import annotations

import numpy as np

from . import evaluate as _ev

__all__ = ["RoadEvaluator"]


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
            return tuple(_out(a, shape) for a in _ev._chunked(lambda a, b: _ev._uv_to_normal(self._p, a, b),
                                                              pu.size, pu, pv))
        z = _ev._chunked(lambda a, b: _ev._uv_to_z(self._p, a, b, grid_only, False), pu.size, pu, pv)
        return _out(z, shape)

    def uv_to_gradient(self, u, v, grid_only: bool = False):
        """Slopes (dz/du, dz/dv) at (u, v). See :meth:`pyopencrg.CrgRoad.uv_to_gradient`."""
        shape, pu, pv = _inputs(u, v)
        _, gu, gv = _ev._chunked(lambda a, b: _ev._uv_to_z(self._p, a, b, grid_only, True), pu.size, pu, pv)
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
        u, v = self._xy_to_uv(px, py, u_hint)
        if with_normal:
            return tuple(_out(a, shape) for a in _ev._chunked(lambda a, b: _ev._uv_to_normal(self._p, a, b),
                                                              u.size, u, v))
        return _out(_ev._chunked(lambda a, b: _ev._uv_to_z(self._p, a, b, False, False), u.size, u, v), shape)

    def xy_to_normal(self, x, y, u_hint=None):
        """Unit normal (nx, ny, nz) at (x, y). See :meth:`pyopencrg.CrgRoad.xy_to_normal`."""
        _, nx, ny, nz = self.xy_to_z(x, y, u_hint, with_normal=True)
        return nx, ny, nz

    def _xy_to_uv(self, px, py, u_hint):
        p = self._p
        if u_hint is not None:
            hint = np.broadcast_to(np.asarray(u_hint, dtype=np.float64), px.shape).ravel()
        elif self._last_u is not None and self._last_u.size == px.size:
            hint = self._last_u
        else:
            hint = None
        if hint is None:
            u, v = _ev._chunked(lambda a, b: _ev._xy_to_uv(p, a, b, None), px.size, px, py)
        else:
            u, v = _ev._chunked(lambda a, b, c: _ev._xy_to_uv(p, a, b, c), px.size, px, py, hint)
            if u_hint is None:  # warm start: check that the previous positions were still close
                bad = ~(np.abs(v) <= self._v_limit)
                if bad.any():
                    u[bad], v[bad] = _ev._xy_to_uv(p, px[bad], py[bad], None)
        self._last_u = u.copy()
        return u, v

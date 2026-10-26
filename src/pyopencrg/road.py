"""
The :class:`CrgRoad` class: one OpenCRG road in memory.

A road is described by:

* a **reference line** (the "center line" of the road): its x/y position,
  its elevation ``zr`` and its banking (cross slope), all sampled every
  ``u_inc`` metres along ``u``;
* an **elevation grid** ``z`` of shape ``(nu, nv)``: ``z[i, j]`` is the height
  of the road surface *relative to the reference line* at distance
  ``u[i]`` along the line and lateral position ``v[j]`` (positive to the left).

The absolute height of a point of the road is therefore::

    z_absolute = z_grid + zr(u) + banking(u) * v

Typical use::

    import pyopencrg as crg

    road = crg.read("my_road.crg")        # read a file
    print(road)                            # summary
    road.check().show()                    # consistency checks
    z = road.uv_to_z(10.0, 0.5)            # height 10 m along, 0.5 m to the left
    small = road.crop(u=(0, 50))           # keep the first 50 m
    small.save("first_50m.crg", fmt="LDFI")
    road.show()                            # overview plots

All editing methods return a **new** road; the original one is not changed.
"""

from __future__ import annotations

import copy
import os

import numpy as np

from . import evaluate as _ev
from .keywords import DEFAULT_MODIFIERS, OPTION_DEFAULTS

__all__ = ["CrgRoad", "read"]


def _as_grid(z):
    """Elevation grid as float64, or float32 if it is already float32 (to save memory)."""
    z = np.asarray(z)
    return z if z.dtype in (np.float32, np.float64) else z.astype(np.float64)


def _as_out(arr, shape):
    """Return a float for scalar inputs, else an array of the broadcast shape."""
    if shape == ():
        return float(np.asarray(arr).reshape(-1)[0])
    return np.asarray(arr).reshape(shape)


def _broadcast(*args):
    arrs = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in args])
    return arrs[0].shape, [np.ascontiguousarray(a).ravel() for a in arrs]


def integrate_heading(x0, y0, phi, u_inc, x_end=None, y_end=None):
    """Positions of the reference line points from the heading angles.

    If an end position is given, the closing error is distributed linearly
    along the line (like the OpenCRG reference implementation does).
    """
    x = np.empty(phi.size + 1)
    y = np.empty(phi.size + 1)
    x[0], y[0] = x0, y0
    np.cumsum(u_inc * np.cos(phi), out=x[1:])
    np.cumsum(u_inc * np.sin(phi), out=y[1:])
    x[1:] += x0
    y[1:] += y0
    if x_end is not None and y_end is not None:
        w = np.linspace(0.0, 1.0, x.size)
        x += w * (x_end - x[-1])
        y += w * (y_end - y[-1])
    return x, y


def integrate_slope(z0, slope, u_inc, z_end=None):
    """Elevation of the reference line from the slope values."""
    zr = np.empty(slope.size + 1)
    zr[0] = z0
    np.cumsum(u_inc * slope, out=zr[1:])
    zr[1:] += z0
    if z_end is not None:
        zr += np.linspace(0.0, 1.0, zr.size) * (z_end - zr[-1])
    return zr


def _make_v(v, nv, v_inc, u_inc):
    if v is None:
        inc = u_inc if v_inc is None else v_inc
        return (np.arange(nv) - (nv - 1) / 2.0) * inc
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if v.size == 1:
        return np.linspace(-abs(v[0]), abs(v[0]), nv)
    if v.size == 2 and nv != 2:
        return np.linspace(v[0], v[1], nv)
    if v.size != nv:
        raise ValueError(f"v must have 1, 2 or nv={nv} values, got {v.size}")
    return v.copy()


class CrgRoad:
    """An OpenCRG road.

    Parameters
    ----------
    z : array_like, shape (nu, nv)
        Elevation grid relative to the reference line (metres). Row ``i`` is
        the cross section at ``u = u_start + i * u_inc``. NaN marks missing data.
    u_inc : float, optional
        Distance between two cross sections (metres). Default 0.01.
    v : array_like, optional
        Lateral positions of the long sections (metres, positive to the left).
        One value = half width of the road; two values = right and left edges;
        nv values = position of each long section. Default: a road centred on
        the reference line with spacing ``v_inc``.
    u_start : float, optional
        u value of the first cross section. Default 0.
    v_inc : float, optional
        Lateral spacing used when ``v`` is not given (default: ``u_inc``).
    heading : float or array_like, optional
        Heading angle of the reference line in radians (0 = along x,
        pi/2 = along y). A single value gives a straight line; an array of
        ``nu - 1`` values gives the heading of each interval.
    slope : float or array_like, optional
        Longitudinal slope of the reference line (m/m, e.g. 0.05 = 5 %). One
        value or ``nu - 1`` values.
    banking : float or array_like, optional
        Banking (cross slope, m/m, positive = left side higher). One value or
        ``nu`` values.
    x_start, y_start, z_start : float, optional
        Position of the start of the reference line.
    comments : list of str, optional
        Free text stored in the ``$CT`` block of the file.

    Examples
    --------
    A flat straight road, 100 m long and 4 m wide, with a 5 cm bump::

        >>> import numpy as np, pyopencrg as crg
        >>> u = np.arange(0, 100.01, 0.05)
        >>> v = np.arange(-2, 2.001, 0.05)
        >>> z = np.zeros((u.size, v.size))
        >>> z[(u > 50) & (u < 50.5), :] = 0.05
        >>> road = crg.CrgRoad(z, u_inc=0.05, v=v)
        >>> road.uv_to_z(50.25, 0.0)
        0.05
    """

    def __init__(self, z, u_inc=0.01, v=None, *, u_start=0.0, v_inc=None, heading=0.0,
                 slope=0.0, banking=0.0, x_start=0.0, y_start=0.0, z_start=0.0, comments=None):
        z = np.array(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2:
            raise ValueError("z must be a 2D array with at least 2 x 2 values")
        nu, nv = z.shape
        if u_inc <= 0:
            raise ValueError("u_inc must be positive")
        phi = np.broadcast_to(np.asarray(heading, dtype=np.float64), (nu - 1,)).astype(np.float64)
        s = np.broadcast_to(np.asarray(slope, dtype=np.float64), (nu - 1,)).astype(np.float64)
        b = np.broadcast_to(np.asarray(banking, dtype=np.float64), (nu,)).astype(np.float64)
        x, y = integrate_heading(float(x_start), float(y_start), phi, float(u_inc))
        zr = integrate_slope(float(z_start), s, float(u_inc))
        self._init_state(z, _make_v(v, nv, v_inc, u_inc), float(u_start), float(u_inc), x, y, zr, b)
        self.comments = list(comments) if comments else []

    # ------------------------------------------------------------------
    # internal construction helpers
    # ------------------------------------------------------------------

    def _init_state(self, z, v, u_start, u_inc, x, y, zr, banking):
        v = np.asarray(v, dtype=np.float64)
        if np.any(np.diff(v) <= 0):
            raise ValueError("v positions must be strictly increasing (from right to left)")
        self._z = z
        self._v = v
        self._u_start = u_start
        self._u_inc = u_inc
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._x.setflags(write=False)
        self._y.setflags(write=False)
        self._zr = np.asarray(zr, dtype=np.float64)
        self._banking = np.asarray(banking, dtype=np.float64)
        self._cache = {}
        #: Free text of the ``$CT`` block.
        self.comments = []
        #: Modifiers (``$ROAD_CRG_MODS``), keyword -> value. ``None`` means that
        #: the file had no modifier block (OpenCRG then uses default modifiers).
        self.mods = None
        #: Options (``$ROAD_CRG_OPTS``) explicitly set, keyword -> value.
        self.opts = {}
        # options of other tools (logging...) found in the file, written back unchanged
        self._other_tool_opts = {}
        #: Map projection (``$ROAD_CRG_MPRO``), keyword -> value, or None.
        self.mpro = None
        #: Other header lines found in the file, written back unchanged.
        self.extra_header = []
        #: Name of the file the road was read from.
        self.filename = ""
        #: Minor problems found while reading the file.
        self.read_warnings = []
        #: Serious problems found while reading the file (truncated or unreadable data).
        self.read_errors = []
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.z_offset = 0.0
        self.phi_offset = 0.0
        self.lon_start = self.lat_start = self.alt_start = None
        self.lon_end = self.lat_end = self.alt_end = None
        # True for a part of a file (its modifiers would be applied relative to the part)
        self._part = False

    @classmethod
    def _from_state(cls, z, v, u_start, u_inc, x, y, zr, banking, template=None):
        obj = cls.__new__(cls)
        obj._init_state(_as_grid(z), v, float(u_start), float(u_inc), x, y, zr, banking)
        if template is not None:
            obj._copy_meta(template)
        return obj

    def _copy_meta(self, other):
        self.comments = list(other.comments)
        self.mods = None if other.mods is None else dict(other.mods)
        self.opts = dict(other.opts)
        self._other_tool_opts = dict(other._other_tool_opts)
        self.mpro = None if other.mpro is None else dict(other.mpro)
        self.extra_header = list(other.extra_header)
        self.filename = other.filename
        self._part = other._part
        for k in ("x_offset", "y_offset", "z_offset", "phi_offset", "lon_start", "lat_start",
                  "alt_start", "lon_end", "lat_end", "alt_end"):
            setattr(self, k, getattr(other, k))

    def _derive(self, z=None, v=None, u_start=None, u_inc=None, x=None, y=None, zr=None, banking=None):
        """New road with some core arrays replaced and the same metadata."""
        return CrgRoad._from_state(
            self._z.copy() if z is None else z,
            self._v.copy() if v is None else v,
            self._u_start if u_start is None else u_start,
            self._u_inc if u_inc is None else u_inc,
            self._x.copy() if x is None else x,
            self._y.copy() if y is None else y,
            self._zr.copy() if zr is None else zr,
            self._banking.copy() if banking is None else banking,
            template=self,
        )

    def copy(self) -> CrgRoad:
        """Independent copy of the road."""
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        new = self._derive()
        new.read_warnings = list(self.read_warnings)
        new.read_errors = list(self.read_errors)
        return new

    # ------------------------------------------------------------------
    # reading / writing
    # ------------------------------------------------------------------

    @classmethod
    def read(cls, path, apply_mods: bool = False, check=True, u=None, step=1, dtype="float64") -> CrgRoad:
        """Read an OpenCRG file. Same as :func:`pyopencrg.read`."""
        return read(path, apply_mods=apply_mods, check=check, u=u, step=step, dtype=dtype)

    def save(self, path, fmt: str = "KRBI") -> None:
        """Write the road to an OpenCRG file.

        Parameters
        ----------
        path : str or path-like
            Name of the file to write (overwritten if it exists).
        fmt : str, optional
            Data format:

            * ``"KRBI"`` (default): binary, float32. Smallest file, read by every tool.
            * ``"KDBI"``: binary, float64 (full precision).
            * ``"LRFI"``: text, single precision.
            * ``"LDFI"``: text, double precision (readable in a text editor).

        Examples
        --------
        >>> road.save("road_binary.crg")             # doctest: +SKIP
        >>> road.save("road_text.crg", fmt="LDFI")   # doctest: +SKIP
        """
        from .io import write_crg_file
        write_crg_file(self, path, fmt)

    # ------------------------------------------------------------------
    # core data
    # ------------------------------------------------------------------

    @property
    def z(self) -> np.ndarray:
        """Elevation grid (nu, nv) relative to the reference line, in metres.

        You can modify it in place, for instance ``road.z += 0.01`` or
        ``road.z[100:200, :] = 0``.
        """
        return self._z

    @z.setter
    def z(self, value):
        value = _as_grid(value)
        if value.shape != self._z.shape:
            raise ValueError(f"new z must have the shape {self._z.shape}, got {value.shape}")
        self._z = value.copy()

    @property
    def nu(self) -> int:
        """Number of cross sections (rows of ``z``)."""
        return self._z.shape[0]

    @property
    def nv(self) -> int:
        """Number of long sections (columns of ``z``)."""
        return self._z.shape[1]

    @property
    def shape(self) -> tuple:
        """Shape of the elevation grid ``(nu, nv)``."""
        return self._z.shape

    @property
    def u_start(self) -> float:
        """u value of the first cross section (m)."""
        return self._u_start

    @property
    def u_inc(self) -> float:
        """Distance between two cross sections (m)."""
        return self._u_inc

    @property
    def u_end(self) -> float:
        """u value of the last cross section (m)."""
        return self._u_start + (self.nu - 1) * self._u_inc

    @property
    def u(self) -> np.ndarray:
        """u value of every cross section (m)."""
        return self._u_start + np.arange(self.nu) * self._u_inc

    @property
    def length(self) -> float:
        """Length of the road along the reference line (m)."""
        return (self.nu - 1) * self._u_inc

    @property
    def v(self) -> np.ndarray:
        """Lateral position of every long section (m, positive to the left)."""
        return self._v

    @property
    def v_inc(self):
        """Lateral spacing (m), or ``None`` if the long sections are not evenly spaced."""
        d = np.diff(self._v)
        if np.ptp(d) <= max(self.get_option("check_eps") * (self._v[-1] - self._v[0]), 1e-12):
            return (self._v[-1] - self._v[0]) / (self.nv - 1)
        return None

    @property
    def v_min(self) -> float:
        """Right edge of the road (m)."""
        return float(self._v[0])

    @property
    def v_max(self) -> float:
        """Left edge of the road (m)."""
        return float(self._v[-1])

    @property
    def width(self) -> float:
        """Width of the road (m)."""
        return float(self._v[-1] - self._v[0])

    @property
    def x(self) -> np.ndarray:
        """x position of the reference line at every cross section (read only)."""
        return self._x

    @property
    def y(self) -> np.ndarray:
        """y position of the reference line at every cross section (read only)."""
        return self._y

    @property
    def zr(self) -> np.ndarray:
        """Elevation of the reference line at every cross section (m)."""
        return self._zr

    @zr.setter
    def zr(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.nu,):
            raise ValueError(f"zr must have {self.nu} values")
        self._zr = value.copy()

    @property
    def banking(self) -> np.ndarray:
        """Banking (cross slope, m/m) at every cross section."""
        return self._banking

    @banking.setter
    def banking(self, value):
        value = np.broadcast_to(np.asarray(value, dtype=np.float64), (self.nu,))
        self._banking = value.astype(np.float64)

    @property
    def phi(self) -> np.ndarray:
        """Heading angle (rad) of each of the ``nu - 1`` intervals of the reference line."""
        if "phi" not in self._cache:
            self._cache["phi"] = np.arctan2(np.diff(self._y), np.diff(self._x))
        return self._cache["phi"]

    @property
    def phi_start(self) -> float:
        """Heading angle at the start of the reference line (rad)."""
        return float(self._mean_heading() if self.is_straight else self.phi[0])

    @property
    def phi_end(self) -> float:
        """Heading angle at the end of the reference line (rad)."""
        return float(self._mean_heading() if self.is_straight else self.phi[-1])

    def _mean_heading(self):
        p = self.phi
        return np.arctan2(np.sin(p).sum(), np.cos(p).sum())

    @property
    def slope(self) -> np.ndarray:
        """Longitudinal slope (m/m) of each of the ``nu - 1`` intervals."""
        return np.diff(self._zr) / self._u_inc

    @property
    def curvature(self) -> np.ndarray:
        """Curvature (1/m) of the reference line at the ``nu - 2`` inner points.

        Positive when the road turns to the left.
        """
        if "rc" not in self._cache:
            dx, dy = np.diff(self._x), np.diff(self._y)
            self._cache["rc"] = (dx[:-1] * dy[1:] - dy[:-1] * dx[1:]) / self._u_inc ** 3
        return self._cache["rc"]

    @property
    def is_straight(self) -> bool:
        """True if the reference line is a straight line."""
        eps = self.get_option("check_eps")
        cached = self._cache.get("straight")
        if cached is None or cached[0] != eps:
            cached = (eps, _straight_line(self._x, self._y, eps))
            self._cache["straight"] = cached
        return cached[1]

    def with_reference_line(self, x, y) -> CrgRoad:
        """New road with another reference line (x/y positions of the ``nu`` points).

        The points should be ``u_inc`` apart; use
        :func:`pyopencrg.generate.reference_line_from_points` to build such a
        line from arbitrary points.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != (self.nu,) or y.shape != (self.nu,):
            raise ValueError(f"x and y must have {self.nu} values")
        return self._derive(x=x.copy(), y=y.copy())

    def surface_z(self) -> np.ndarray:
        """Absolute elevation at every grid node (grid + reference line elevation + banking)."""
        return self._z + self._zr[:, None] + self._banking[:, None] * self._v[None, :]

    def grid_xy(self, step: int = 1):
        """x and y positions of every grid node, arrays of shape (nu, nv).

        Parameters
        ----------
        step : int, optional
            Keep only one node out of ``step`` in both directions (to save memory).
        """
        uu, vv = np.meshgrid(self.u[::step], self._v[::step], indexing="ij")
        x, y = self.uv_to_xy(uu, vv)
        return x, y

    # ------------------------------------------------------------------
    # options and modifiers
    # ------------------------------------------------------------------

    def get_option(self, name: str):
        """Value of an option (``$ROAD_CRG_OPTS``), or its default value.

        Examples
        --------
        >>> road.get_option("border_mode_u")   # doctest: +SKIP
        2
        """
        name = name.lower()
        if name in self.opts:
            return self.opts[name]
        if name == "check_tol":
            return 0.1 * self.get_option("check_inc")
        if name not in OPTION_DEFAULTS:
            raise KeyError(f"unknown option '{name}'")
        return OPTION_DEFAULTS[name]

    def set_option(self, name: str, value) -> None:
        """Set an option (``$ROAD_CRG_OPTS``), e.g. ``road.set_option("border_mode_u", 0)``."""
        name = name.lower()
        if name not in OPTION_DEFAULTS:
            raise KeyError(f"unknown option '{name}', valid options: {sorted(OPTION_DEFAULTS)}")
        if name in ("border_mode_u", "border_mode_v") and value not in (0, 1, 2, 3, 4):
            raise ValueError(f"{name} must be 0, 1, 2, 3 or 4")
        if name == "refline_continuation" and value not in (0, 1):
            raise ValueError("refline_continuation must be 0 or 1")
        self.opts[name] = value
        self._cache.pop("closure", None)
        self._cache.pop("straight", None)

    def effective_mods(self) -> dict:
        """Modifiers that :meth:`apply_mods` would apply (defaults if the file had none)."""
        return dict(DEFAULT_MODIFIERS) if self.mods is None else dict(self.mods)

    def apply_mods(self) -> CrgRoad:
        """Apply the modifiers (``$ROAD_CRG_MODS``) and return the modified road.

        Modifiers scale, fill NaN values or move the road. Simulators apply
        them when they load a file. After this call the returned road has an
        empty modifier list, so they will not be applied twice.

        If the file had no modifier block, the OpenCRG default modifiers are
        used: NaN values are replaced by the last valid value of the cross
        section, and the start of the reference line is moved to x = y = z = 0
        with heading 0.

        On a part of a file (``crg.read(path, u=...)``), the road would be
        moved relative to the part, not to the whole file. Read the part with
        ``apply_mods=True`` instead (``crg.read(path, u=..., apply_mods=True)``
        or ``f.read(u=..., apply_mods=True)``): it is then placed exactly
        where it is in the modified whole road.
        """
        from .mods import apply_mods
        return apply_mods(self)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def uv_to_z(self, u, v, grid_only: bool = False, with_normal: bool = False):
        """Elevation of the road at road coordinates (u, v).

        Parameters
        ----------
        u, v : float or array_like
            Distance along the reference line and lateral position (m).
            Arrays are broadcast together (e.g. a column and a row give a grid).
        grid_only : bool, optional
            If True, return only the elevation grid value (without reference
            line elevation and banking).
        with_normal : bool, optional
            If True, also return the unit normal vector of the surface (see
            :meth:`uv_to_normal`), computed in the same pass.

        Returns
        -------
        z : float or numpy.ndarray
            Elevation (m). NaN where the road has no data.
        nx, ny, nz : float or numpy.ndarray
            Only with ``with_normal=True``.

        Examples
        --------
        >>> road.uv_to_z(12.5, 0.0)                           # doctest: +SKIP
        >>> road.uv_to_z(np.linspace(0, 100, 1001), 0.9)      # long profile at v = 0.9 m  # doctest: +SKIP
        >>> z, nx, ny, nz = road.uv_to_z(12.5, 0.0, with_normal=True)   # doctest: +SKIP
        """
        shape, (pu, pv) = _broadcast(u, v)
        if with_normal:
            if grid_only:
                raise ValueError("with_normal=True is not available with grid_only=True")
            return tuple(_as_out(a, shape) for a in _ev.uv_to_normal(self, pu, pv))
        return _as_out(_ev.uv_to_z(self, pu, pv, grid_only), shape)

    def uv_to_gradient(self, u, v, grid_only: bool = False):
        """Slope of the surface along u and along v at road coordinates (u, v).

        The derivatives include the reference line elevation and the
        banking (unless ``grid_only=True``). They are exact for the bilinear
        interpolation used by :meth:`uv_to_z`.

        Returns
        -------
        dz_du, dz_dv : float or numpy.ndarray
            Slopes in m/m (0.05 = 5 %).
        """
        shape, (pu, pv) = _broadcast(u, v)
        _, gu, gv = _ev.uv_to_z_gradient(self, pu, pv, grid_only)
        return _as_out(gu, shape), _as_out(gv, shape)

    def uv_to_normal(self, u, v):
        """Unit normal vector of the road surface at road coordinates (u, v).

        The normal points upwards and is given in the inertial x/y/z frame:
        ``(0, 0, 1)`` on a flat horizontal road. It takes into account the
        elevation grid, the slope, the banking and the curvature of the
        road. This is the direction of the normal contact force of a tyre.

        Returns
        -------
        nx, ny, nz : float or numpy.ndarray
        """
        shape, (pu, pv) = _broadcast(u, v)
        _, nx, ny, nz = _ev.uv_to_normal(self, pu, pv)
        return _as_out(nx, shape), _as_out(ny, shape), _as_out(nz, shape)

    def uv_to_xy(self, u, v):
        """Inertial position (x, y) of road coordinates (u, v).

        Returns
        -------
        x, y : float or numpy.ndarray
        """
        shape, (pu, pv) = _broadcast(u, v)
        x, y = _ev.uv_to_xy(self, pu, pv)
        return _as_out(x, shape), _as_out(y, shape)

    def xy_to_uv(self, x, y, u_hint=None):
        """Road coordinates (u, v) of inertial positions (x, y).

        Parameters
        ----------
        x, y : float or array_like
            Positions in the inertial system.
        u_hint : float or array_like, optional
            Approximate u of the points: one value for all of them (e.g. the
            u of the vehicle) or one value per point. The search is faster,
            and it is needed for roads that pass several times near the same
            place (crossings, overpasses).

        Returns
        -------
        u, v : float or numpy.ndarray
        """
        shape, (px, py) = _broadcast(x, y)
        hint = None if u_hint is None else np.broadcast_to(np.asarray(u_hint, float), shape).ravel()
        u, v = _ev.xy_to_uv(self, px, py, hint)
        return _as_out(u, shape), _as_out(v, shape)

    def xy_to_z(self, x, y, u_hint=None, with_normal: bool = False):
        """Elevation of the road at inertial positions (x, y).

        Parameters
        ----------
        u_hint : float or array_like, optional
            Approximate u of the points, see :meth:`xy_to_uv`.
        with_normal : bool, optional
            If True, also return the unit normal vector (see :meth:`uv_to_normal`).

        Returns
        -------
        z : float or numpy.ndarray
        nx, ny, nz : float or numpy.ndarray
            Only with ``with_normal=True``.
        """
        shape, (px, py) = _broadcast(x, y)
        hint = None if u_hint is None else np.broadcast_to(np.asarray(u_hint, float), shape).ravel()
        u, v = _ev.xy_to_uv(self, px, py, hint)
        if with_normal:
            return tuple(_as_out(a, shape) for a in _ev.uv_to_normal(self, u, v))
        return _as_out(_ev.uv_to_z(self, u, v), shape)

    def xy_to_normal(self, x, y, u_hint=None):
        """Unit normal vector (nx, ny, nz) of the road surface at inertial positions (x, y).

        See :meth:`uv_to_normal` and, for ``u_hint``, :meth:`xy_to_uv`.
        """
        _, nx, ny, nz = self.xy_to_z(x, y, u_hint, with_normal=True)
        return nx, ny, nz

    def uv_to_xyz(self, u, v):
        """Inertial position (x, y, z) of road coordinates (u, v)."""
        x, y = self.uv_to_xy(u, v)
        return x, y, self.uv_to_z(u, v)

    def evaluator(self):
        """Fast evaluator for repeated calls with few points (e.g. a simulation loop).

        The evaluator prepares everything once and has the same evaluation
        methods as the road (``uv_to_z``, ``xy_to_z``, ``xy_to_uv``,
        ``uv_to_normal``...), with a lower cost per call. It also remembers
        the last u found for each point and uses it as the starting point of
        the next search: points that move continuously (tyre contact
        points) are found almost immediately.

        Create it again if you replace an array of the road (``road.z = ...``)
        or change an option. Changing the values of ``road.z`` in place is
        seen by the evaluator.

        Examples
        --------
        >>> ev = road.evaluator()                                   # doctest: +SKIP
        >>> for step in range(n_steps):                             # doctest: +SKIP
        ...     z, nx, ny, nz = ev.xy_to_z(x, y, with_normal=True)
        """
        from .evaluator import RoadEvaluator
        return RoadEvaluator(self)

    def u_to_phi(self, u):
        """Heading angle (rad) of the reference line at u."""
        shape, (pu,) = _broadcast(u)
        return _as_out(_ev.u_to_phi(self, pu), shape)

    def u_to_curvature(self, u):
        """Curvature (1/m) of the reference line at u (positive = turning left)."""
        shape, (pu,) = _broadcast(u)
        return _as_out(_ev.u_to_curvature(self, pu), shape)

    def uv_to_index(self, u, v=None):
        """Grid indices (0 based) of the cell containing (u, v).

        Returns ``iu`` if only ``u`` is given, otherwise ``(iu, iv)``.
        """
        if v is None:
            shape, (pu,) = _broadcast(u)
            iu = _ev.uv_to_index(self, pu)
            return int(iu[0]) if shape == () else iu.reshape(shape)
        shape, (pu, pv) = _broadcast(u, v)
        iu, iv = _ev.uv_to_index(self, pu, pv)
        if shape == ():
            return int(iu[0]), int(iv[0])
        return iu.reshape(shape), iv.reshape(shape)

    def cross_section(self, u, absolute: bool = True):
        """Cross section at distance ``u``: returns ``(v, z)`` arrays."""
        z = self.uv_to_z(np.full(self.nv, float(u)), self._v, grid_only=not absolute)
        return self._v.copy(), z

    def long_section(self, v=0.0, absolute: bool = True):
        """Long section (longitudinal profile) at lateral position ``v``: returns ``(u, z)``."""
        u = self.u
        return u, self.uv_to_z(u, np.full(u.size, float(v)), grid_only=not absolute)

    # --- geodetic transformations ------------------------------------

    def xyz_to_enh(self, x, y, z):
        """Local inertial (x, y, z) -> projected map coordinates (east, north, height).

        Uses ``reference_line_offset_x/y/z/phi`` of the road.
        """
        from .mapping import xyz_to_enh
        return xyz_to_enh(self, x, y, z)

    def enh_to_xyz(self, e, n, h):
        """Projected map coordinates (east, north, height) -> local inertial (x, y, z)."""
        from .mapping import enh_to_xyz
        return enh_to_xyz(self, e, n, h)

    def xyz_to_llh(self, x, y, z):
        """Local (x, y, z) -> WGS84 (latitude, longitude in degrees, altitude).

        Requires a map projection (``road.mpro``).
        """
        from .mapping import xyz_to_llh
        return xyz_to_llh(self, x, y, z)

    def llh_to_xyz(self, lat, lon, alt):
        """WGS84 (latitude, longitude in degrees, altitude) -> local (x, y, z).

        Requires a map projection (``road.mpro``).
        """
        from .mapping import llh_to_xyz
        return llh_to_xyz(self, lat, lon, alt)

    def xy_to_wgs84(self, x, y):
        """Local (x, y) -> WGS84 (latitude, longitude) in degrees.

        Uses the map projection if available, otherwise the WGS84 start and
        end points of the header.
        """
        from .wgs84 import road_xy_to_wgs84
        return road_xy_to_wgs84(self, x, y)

    def set_wgs84_end(self, x_axis_azimuth: float = 90.0) -> None:
        """Compute the WGS84 end point from the start point.

        Parameters
        ----------
        x_axis_azimuth : float, optional
            Direction of the x axis, in degrees clockwise from north
            (default 90: x points to the east).
        """
        from .wgs84 import set_wgs84_end
        set_wgs84_end(self, x_axis_azimuth)

    # ------------------------------------------------------------------
    # checks and information
    # ------------------------------------------------------------------

    def check(self):
        """Run all consistency checks and return a :class:`pyopencrg.checks.CheckReport`.

        Examples
        --------
        >>> report = road.check()      # doctest: +SKIP
        >>> report.ok                  # doctest: +SKIP
        True
        >>> print(report)              # doctest: +SKIP
        """
        from .checks import check
        return check(self)

    def info(self) -> str:
        """Readable summary of the road (also printed by ``print(road)``)."""
        from .checks import info_text
        return info_text(self)

    def __repr__(self):
        name = os.path.basename(self.filename) if self.filename else "unsaved"
        kind = "straight" if self.is_straight else "curved"
        return (f"<CrgRoad '{name}': {self.length:.3f} m x {self.width:.3f} m, grid {self.nu} x {self.nv}, "
                f"u_inc={self.u_inc:g} m, {kind} reference line>")

    def __str__(self):
        return self.info()

    def stats(self) -> dict:
        """Basic statistics of the elevation grid (min, max, mean, std, NaN ratio)."""
        from .analysis import stats
        return stats(self)

    def compare(self, other):
        """Compare with another road, see :func:`pyopencrg.compare`."""
        from .edit import compare
        return compare(self, other)

    # ------------------------------------------------------------------
    # editing (all return a new road)
    # ------------------------------------------------------------------

    def crop(self, u=None, v=None) -> CrgRoad:
        """Keep only a part of the road, given in metres.

        Parameters
        ----------
        u : (float, float), optional
            First and last u to keep (m). ``None`` for one of them = no limit.
        v : (float, float), optional
            Right and left lateral limits to keep (m).

        Examples
        --------
        >>> part = road.crop(u=(100, 200))            # doctest: +SKIP
        >>> lane = road.crop(v=(-1.75, 1.75))         # doctest: +SKIP
        """
        from .edit import crop
        return crop(self, u, v)

    def crop_index(self, iu=None, iv=None) -> CrgRoad:
        """Keep only a part of the grid, given by indices ``(first, last)`` (both included, 0 based)."""
        from .edit import crop_index
        return crop_index(self, iu, iv)

    def trim(self, before: float = 0.0, after: float = 0.0) -> CrgRoad:
        """Remove ``before`` metres at the start and ``after`` metres at the end."""
        from .edit import trim
        return trim(self, before, after)

    def trim_nan_borders(self) -> CrgRoad:
        """Remove the cross sections at the start and the end that contain NaN values."""
        from .edit import trim_nan_borders
        return trim_nan_borders(self)

    def add_ramps(self, before: float = 2.0, after: float = 2.0) -> CrgRoad:
        """Add smooth access ramps before the start and after the end of the road.

        The added cross sections blend linearly from the mean height of the
        first (last) cross section to its real profile.
        """
        from .edit import add_ramps
        return add_ramps(self, before, after)

    def append(self, other: CrgRoad) -> CrgRoad:
        """Append another road at the end of this one (see :func:`pyopencrg.concatenate`)."""
        from .edit import append
        return append(self, other)

    def flip(self) -> CrgRoad:
        """Reverse the driving direction: start and end, left and right are swapped."""
        from .edit import flip
        return flip(self)

    def filter(self, method: str = "mean", size=(3, 3), weight: float = 1.0, repeat: int = 1,
               iu=None, iv=None) -> CrgRoad:
        """Filter the elevation grid with a small convolution mask.

        Parameters
        ----------
        method : str
            ``"mean"`` (moving average), ``"gauss"`` (binomial smoothing),
            ``"sobel"`` (first derivative along u), ``"2diff"`` (second
            derivative along u) or ``"laplace"`` (second derivatives).
        size : int or (int, int)
            Size of the mask along u and v (number of grid points).
        weight : float
            Factor applied to the mask.
        repeat : int
            Number of times the filter is applied.
        iu, iv : (int, int), optional
            Restrict the filter to this part of the grid (indices, included).
        """
        from .edit import filter_grid
        return filter_grid(self, method, size, weight, repeat, iu, iv)

    def clip_z(self, z_min=None, z_max=None, iu=None, iv=None, absolute: bool = True) -> CrgRoad:
        """Limit the elevation between ``z_min`` and ``z_max``.

        By default the absolute elevation is limited; use ``absolute=False``
        to limit only the grid values.
        """
        from .edit import clip_z
        return clip_z(self, z_min, z_max, iu, iv, absolute)

    def resample(self, u_inc=None, v_inc=None, v=None) -> CrgRoad:
        """Re-sample the road on a new grid.

        Parameters
        ----------
        u_inc : float, optional
            New distance between cross sections (default: unchanged).
        v_inc : float, optional
            New lateral spacing (default: unchanged).
        v : float, (float, float) or array, optional
            New lateral extent: half width, right/left edges or positions.
        """
        from .edit import resample
        return resample(self, u_inc, v_inc, v)

    def set_banking(self, banking=0.0) -> CrgRoad:
        """Change the banking without changing the road surface.

        The elevation grid is corrected so that the absolute surface stays
        the same. ``road.set_banking(0)`` merges the banking into the grid.
        """
        from .edit import set_banking
        return set_banking(self, banking)

    def set_reference_elevation(self, zr=None) -> CrgRoad:
        """Change the reference line elevation without changing the road surface.

        Parameters
        ----------
        zr : float, (float, float) or array of nu values, optional
            New elevation: constant, linear between start and end values, or
            one value per cross section. Default: the mean elevation (the
            slope is merged into the grid).
        """
        from .edit import set_reference_elevation
        return set_reference_elevation(self, zr)

    def set_slope(self, slope) -> CrgRoad:
        """Change the longitudinal slope without changing the road surface."""
        from .edit import set_slope
        return set_slope(self, slope)

    def extract_slope(self, smoothing: float = 0.01) -> CrgRoad:
        """Move the longitudinal slope from the grid into the reference line.

        Parameters
        ----------
        smoothing : float in [0, 1]
            0 = straight line fit, 1 = follow the profile exactly.
        """
        from .edit import extract_slope
        return extract_slope(self, smoothing)

    def extract_banking(self, smoothing: float = 0.01) -> CrgRoad:
        """Move the banking (cross slope) from the grid into the reference line.

        Parameters
        ----------
        smoothing : float in [0, 1]
            0 = straight line fit along u, 1 = one banking value per cross section.
        """
        from .edit import extract_banking
        return extract_banking(self, smoothing)

    def separate_slope_banking(self, slope_window: float = 0.0, banking_window: float = 0.0) -> CrgRoad:
        """Find slope and banking, filter them and separate them from the grid.

        Parameters
        ----------
        slope_window, banking_window : float
            Filter window length (m). Special values: ``0`` = do not separate,
            ``-1`` = separate without filtering, ``-2`` = constant
            (linear slope / mean banking).
        """
        from .edit import separate_slope_banking
        return separate_slope_banking(self, slope_window, banking_window)

    def overlay_uv(self, other: CrgRoad, iu=None, iv=None) -> CrgRoad:
        """Add the elevation of another road, matched in road coordinates (u, v).

        The other road is re-sampled to the same increments and repeated if
        it is smaller (useful to add a texture or an obstacle).
        """
        from .edit import overlay_uv
        return overlay_uv(self, other, iu, iv)

    def overlay_xy(self, other: CrgRoad, iu=None, iv=None, outside=None) -> CrgRoad:
        """Add the elevation of another road, matched in inertial coordinates (x, y).

        Parameters
        ----------
        outside : float, optional
            Value used where the points fall outside of ``other`` (default:
            the border mode of ``other`` decides).
        """
        from .edit import overlay_xy
        return overlay_xy(self, other, iu, iv, outside)

    def fill_nan(self, mode: int = 2, offset: float = 0.0) -> CrgRoad:
        """Replace NaN values of the grid.

        Parameters
        ----------
        mode : int
            1 = set to ``offset``; 2 = keep the last valid value of the cross
            section (plus ``offset``). NaN values between valid values are
            interpolated linearly along the cross section.
        """
        from .edit import fill_nan
        return fill_nan(self, mode, offset)

    def fill_holes(self, method: str = "linear", margin: int = 5, max_points: int = 40000) -> CrgRoad:
        """Fill the NaN holes of the grid by 2D interpolation (only around each hole).

        Parameters
        ----------
        method : str
            ``"linear"``, ``"cubic"``, ``"nearest"`` or ``"rbf"`` (radial basis functions).
        margin : int
            Number of valid grid points around each hole used for the interpolation.
        max_points : int
            For ``"rbf"``: maximum number of known points (they are sub-sampled above).
        """
        from .analysis import fill_holes
        return fill_holes(self, method, margin, max_points)

    def remove_artifacts(self, normal_tol: float = 0.3, radius: float = 15) -> CrgRoad:
        """Set to NaN the scanning artifacts (very steep points) and their surroundings.

        Parameters
        ----------
        normal_tol : float
            Points whose normal vector has a vertical component below this
            value (very steep) are considered as artifacts.
        radius : float
            Radius around each artifact, in grid points, that is also removed.
        """
        from .analysis import remove_artifacts
        return remove_artifacts(self, normal_tol, radius)

    def add_profile(self, u_section, u_profile, v_section, v_profile, operation: str = "add") -> CrgRoad:
        """Add (or multiply) a profile defined along u and along v.

        The value added at (u, v) is ``fu(u) * fv(v)`` where ``fu`` and ``fv``
        are piecewise linear functions given by points, and 0 outside of them.

        Examples
        --------
        A 2 cm high, 20 cm wide road marking at v = 1.75 m along the whole road::

            >>> marked = road.add_profile([road.u_start, road.u_end], [1, 1],
            ...                           [1.64, 1.65, 1.85, 1.86], [0, 0.02, 0.02, 0])  # doctest: +SKIP
        """
        from .edit import add_profile
        return add_profile(self, u_section, u_profile, v_section, v_profile, operation)

    def add_random(self, u_section, u_profile, v_section, v_profile, seed=0, amp_min=0.0,
                   amp_max=0.01, u_smooth=0.0, v_smooth=0.0, operation: str = "add") -> CrgRoad:
        """Add a random roughness, weighted by profiles along u and v (see :meth:`add_profile`)."""
        from .edit import add_random
        return add_random(self, u_section, u_profile, v_section, v_profile, seed, amp_min, amp_max,
                          u_smooth, v_smooth, operation)

    def engrave_image(self, image_path, height: float = 0.01, width: float = 1.0, u0=None, v0=None,
                      rotate: bool = False, mirror: bool = False, threshold: int = 127) -> CrgRoad:
        """Add a black and white image (logo, text...) on the road surface.

        White pixels raise the surface by ``height`` metres.

        Parameters
        ----------
        image_path : str
            Image file (PNG, JPG...). Requires the Pillow package.
        height : float
            Height added where the image is white (m, can be negative).
        width : float
            Size of the image along u (m).
        u0, v0 : float, optional
            Position of the image corner (default: start of the road, right edge).
        rotate : bool
            Rotate the image by 90 degrees.
        mirror : bool
            Mirror the image.
        """
        from .edit import engrave_image
        return engrave_image(self, image_path, height, width, u0, v0, rotate, mirror, threshold)

    def move_to(self, x=0.0, y=0.0, z=0.0, phi=0.0, u=None, v=0.0) -> CrgRoad:
        """Move and rotate the road so that a reference point gets a given position.

        Parameters
        ----------
        x, y, z, phi : float
            Target position and heading (rad) of the reference point.
        u, v : float
            Reference point on the road (default: start of the reference line).
        """
        from .mods import move_to
        return move_to(self, x, y, z, phi, u, v)

    def transform(self, dx=0.0, dy=0.0, dz=0.0, dphi=0.0, center=None) -> CrgRoad:
        """Rotate the road by ``dphi`` (rad) around ``center`` then translate it.

        ``center`` defaults to the start of the reference line.
        """
        from .mods import transform
        return transform(self, dx, dy, dz, dphi, center)

    def scale(self, z=1.0, slope=1.0, banking=1.0, length=1.0, width=1.0, curvature=1.0) -> CrgRoad:
        """Scale the elevation, slope, banking, length, width or curvature of the road."""
        from .mods import scale
        return scale(self, z, slope, banking, length, width, curvature)

    def as_single(self) -> CrgRoad:
        """Round all core data to single precision (float32), like a KRBI file would."""
        new = self._derive(z=self._z.astype(np.float32).astype(np.float64))
        new._banking = self._banking.astype(np.float32).astype(np.float64)
        return new

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------

    def normal_map(self) -> np.ndarray:
        """Vertical component of the surface normal at every grid node (1 = flat, 0 = vertical)."""
        from .analysis import normal_map
        return normal_map(self)

    def find_peaks(self, threshold: float = 0.5, size: int = 3, iu=None, iv=None):
        """Detect peaks (isolated high points) in the grid, see :func:`pyopencrg.analysis.find_peaks`."""
        from .analysis import find_peaks
        return find_peaks(self, threshold, size, iu, iv)

    def nan_regions(self, margin: int = 50):
        """Rectangles ``(iu0, iu1, iv0, iv1)`` around the groups of NaN values of the grid."""
        from .analysis import nan_regions
        return nan_regions(self._z, margin)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def to_obj(self, path, step: int = 1, u=None) -> None:
        """Export the road surface as a Wavefront OBJ mesh (3D software, game engines)."""
        from .export import to_obj
        to_obj(self, path, step, u)

    def to_stl(self, path, step: int = 1, u=None) -> None:
        """Export the road surface as a binary STL mesh."""
        from .export import to_stl
        to_stl(self, path, step, u)

    def to_csv(self, path, step: int = 1) -> None:
        """Export the grid nodes (u, v, x, y, z) to a CSV file (Excel, ...)."""
        from .export import to_csv
        to_csv(self, path, step)

    def to_trm(self, path, step: float = 1.0, v: float = 0.0, extrapolation: float = 0.0,
               z: str = "zero", offset_file: bool = True) -> None:
        """Export the reference line as a Simpack trajectory (``.trm``) file."""
        from .export import to_trm
        to_trm(self, path, step, v, extrapolation, z, offset_file)

    def save_heightmap(self, path, cmap: str = "Greys", absolute: bool = False) -> None:
        """Save the elevation grid as an image (PNG...)."""
        from .export import save_heightmap
        save_heightmap(self, path, cmap, absolute)

    def save_normalmap(self, path, cmap: str = "Blues") -> None:
        """Save the normal map (see :meth:`normal_map`) as an image."""
        from .export import save_normalmap
        save_normalmap(self, path, cmap)

    def save_combined_map(self, path, n_slices: int = 1) -> None:
        """Save normal map and height map stacked in one image, optionally cut in slices."""
        from .export import save_combined_map
        save_combined_map(self, path, n_slices)

    def to_html_map(self, path, n_points: int = 2000) -> None:
        """Write an HTML page showing the road on an OpenStreetMap map (needs WGS84 data)."""
        from .export import to_html_map
        to_html_map(self, path, n_points)

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------

    def show(self, iu=None, iv=None, block: bool = True):
        """Show the standard set of overview figures (like ``crg_show`` in MATLAB).

        Returns the list of matplotlib figures.
        """
        from .plot import show
        return show(self, iu, iv, block=block)

    def view3d(self, color: str = "z", z_scale: float = 1.0, **kwargs):
        """Interactive 3D view with PyVista (fast, handles millions of points).

        Needs ``pip install "pyopencrg[viewer]"``. See :func:`pyopencrg.viewer.view3d`.

        Parameters
        ----------
        color : str
            ``"z"`` (absolute height), ``"grid"`` (grid height) or ``"normal"`` (normal map).
        z_scale : float
            Vertical exaggeration.
        """
        from .viewer import view3d
        return view3d(self, color=color, z_scale=z_scale, **kwargs)

    @property
    def plot(self):
        """Access to all plot functions, e.g. ``road.plot.heightmap()``.

        See :mod:`pyopencrg.plot` for the list of available plots.
        """
        from .plot import PlotAccessor
        return PlotAccessor(self)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _reference_line(c, nu):
    """Reference line of all the ``nu`` records of a file: (x, y, zr, banking, warnings)."""
    head = c.head
    warns = list(c.warnings)
    uinc = c.u_inc
    opts = dict(c.opts)
    ceps = opts.get("check_eps", OPTION_DEFAULTS["check_eps"])
    ctol = opts.get("check_tol", 0.1 * opts.get("check_inc", OPTION_DEFAULTS["check_inc"]))
    length = (nu - 1) * uinc

    if "reference_line_end_u" in head:
        nexp = round((head["reference_line_end_u"] - c.u_start) / uinc) + 1
        if nexp != nu:
            warns.append(f"header u range gives {nexp} cross sections but the file has {nu}")

    x0 = head.get("reference_line_start_x", 0.0)
    y0 = head.get("reference_line_start_y", 0.0)
    xe = head.get("reference_line_end_x")
    ye = head.get("reference_line_end_y")
    if c.phi is not None:
        phi = c.phi
        if np.any(np.isnan(phi)):
            warns.append("NaN heading values replaced by the previous value")
            phi = _ffill(phi)
        x, y = integrate_heading(x0, y0, phi, uinc)
        if xe is not None and ye is not None:
            err = np.hypot(x[-1] - xe, y[-1] - ye)
            if err > max(length * ceps, ctol):
                warns.append(f"reference line end position differs from the integrated headings by {err:.4g} m")
            x, y = integrate_heading(x0, y0, phi, uinc, xe, ye)
    else:
        pbeg = head.get("reference_line_start_phi", 0.0)
        pend = head.get("reference_line_end_phi", pbeg)
        if (np.cos(pend) - np.cos(pbeg)) ** 2 + (np.sin(pend) - np.sin(pbeg)) ** 2 > (np.pi * ceps) ** 2:
            warns.append("no heading channel but start and end headings differ: start heading used")
        phi = np.full(nu - 1, pbeg)
        x, y = integrate_heading(x0, y0, phi, uinc)
        if xe is not None and ye is not None:
            err = np.hypot(x[-1] - xe, y[-1] - ye)
            if err > max(length * ceps, ctol):
                warns.append(f"reference line end position inconsistent with the start position and heading ({err:.4g} m)")

    z0 = head.get("reference_line_start_z", 0.0)
    ze = head.get("reference_line_end_z")
    if c.slope is not None:
        s = c.slope
        if np.any(np.isnan(s)):
            warns.append("NaN slope values replaced by the previous value")
            s = _ffill(s)
        zr = integrate_slope(z0, s, uinc)
        if ze is not None:
            if abs(zr[-1] - ze) > max(length * ceps, ctol):
                warns.append(f"reference line end elevation differs from the integrated slope by {abs(zr[-1] - ze):.4g} m")
            zr = integrate_slope(z0, s, uinc, ze)
    else:
        sbeg = head.get("reference_line_start_s", 0.0)
        send = head.get("reference_line_end_s", sbeg)
        if abs(send - sbeg) > ceps:
            warns.append("no slope channel but start and end slopes differ: start slope used")
        zr = integrate_slope(z0, np.full(nu - 1, sbeg), uinc)
        if ze is not None and abs(zr[-1] - ze) > max(length * ceps, ctol):
            warns.append("reference line end elevation inconsistent with start elevation and slope")

    if c.banking is not None:
        b = c.banking
        if np.any(np.isnan(b)):
            warns.append("NaN banking values replaced by the previous value")
            b = _ffill(b)
    else:
        bbeg = head.get("reference_line_start_b", 0.0)
        if abs(head.get("reference_line_end_b", bbeg) - bbeg) > ceps:
            warns.append("no banking channel but start and end banking differ: start banking used")
        b = np.full(nu, bbeg)
    return x, y, zr, b, warns


def _straight_line(x, y, eps):
    """True if the line (x, y) is straight (see :attr:`CrgRoad.is_straight`)."""
    return bool(np.ptp(np.unwrap(np.arctan2(np.diff(y), np.diff(x)))) <= eps * np.pi)


def _set_part_line(road, x, y, rows, straight):
    """Give a part (cross sections ``rows`` of the line x, y) the geometry of the whole line.

    A part is straight only if the whole line is, and the normals at its first
    and last cross sections are the ones of the whole line, so that parts
    evaluate exactly like the whole road.
    """
    road._cache["straight"] = (road.get_option("check_eps"), bool(straight))
    if not straight:
        from .evaluate import _line_normals
        step = int(rows[1] - rows[0])
        a, b = int(rows[0] > 0), int(rows[-1] + step < x.size)
        ext = np.arange(rows[0] - a * step, rows[-1] + (1 + b) * step, step)
        n1, n2 = _line_normals(x[ext], y[ext])
        road._cache["normals"] = (n1[a:n1.shape[0] - b], n2[a:n2.shape[0] - b])


def _assemble(c, z, v, rows, x, y, zr, b, warns, straight=None):
    """Road made of the grid ``z`` (records ``rows`` of the file) and the reference line of the file.

    ``straight``: straightness of the whole line, if already known.
    """
    nu = x.size
    head = c.head
    uinc = c.u_inc
    if rows is not None and rows.size != nu:  # partial or decimated reading: keep the selected cross sections
        step = int(rows[1] - rows[0])
        road = CrgRoad._from_state(z, v, c.u_start + rows[0] * uinc, uinc * step, x[rows], y[rows], zr[rows], b[rows])
        road._part = True
    else:
        road = CrgRoad._from_state(z, v, c.u_start, uinc, x.copy(), y.copy(), zr.copy(), b.copy())
    road.comments = list(c.comments)
    road.opts = dict(c.opts)
    road._other_tool_opts = dict(c.other_tool_opts)
    road.mods = None if c.mods is None else dict(c.mods)
    road.mpro = None if c.mpro is None else dict(c.mpro)
    road.extra_header = list(c.extra)
    road.filename = c.filename
    road.x_offset = head.get("reference_line_offset_x", 0.0)
    road.y_offset = head.get("reference_line_offset_y", 0.0)
    road.z_offset = head.get("reference_line_offset_z", 0.0)
    road.phi_offset = head.get("reference_line_offset_phi", 0.0)
    road.lon_start = head.get("reference_line_start_lon")
    road.lat_start = head.get("reference_line_start_lat")
    road.alt_start = head.get("reference_line_start_alt")
    road.lon_end = head.get("reference_line_end_lon")
    road.lat_end = head.get("reference_line_end_lat")
    road.alt_end = head.get("reference_line_end_alt")
    if road.alt_start is not None and road.alt_end is None:
        road.alt_end = road.alt_start + (zr[-1] - zr[0])
    road.read_warnings = list(warns)
    road.read_errors = list(c.errors)
    if road._part:
        if straight is None:
            straight = _straight_line(x, y, road.get_option("check_eps"))
        _set_part_line(road, x, y, rows, straight)
    return road


def _road_from_content(c) -> CrgRoad:
    nu = c.n_records or c.z.shape[0]  # the reference line is built on all records of the file
    x, y, zr, b, warns = _reference_line(c, nu)
    return _assemble(c, c.z, c.v, c.rows, x, y, zr, b, warns)


def _grid_dtype(dtype):
    dt = np.dtype(dtype)
    if dt not in (np.float32, np.float64):
        raise ValueError("dtype must be 'float64' or 'float32'")
    return dt


def _ffill(a):
    a = a.copy()
    mask = np.isnan(a)
    if mask.all():
        return np.zeros_like(a)
    idx = np.where(~mask, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    first = np.flatnonzero(~np.isnan(a))[0]
    a[:first] = a[first]
    return a


def read(path, apply_mods: bool = False, check=True, u=None, step=1, dtype="float64") -> CrgRoad:
    """Read an OpenCRG file.

    The file is checked when it is read (see :meth:`CrgRoad.check`): a file
    with errors (truncated data, cross sections without any value, illegal
    options...) is refused, and the warnings are shown once.

    Parameters
    ----------
    path : str or path-like
        The ``.crg`` file (any of the KRBI, KDBI, LRFI, LDFI formats).
    apply_mods : bool, optional
        If True, the modifiers of the file are applied right away (like a
        simulator does). By default the road is returned exactly as stored
        in the file, and you can apply them later with ``road.apply_mods()``.
        With ``u`` or ``step``, the modifiers are computed on the whole file:
        the part is placed exactly where it is in the modified whole road.
    check : bool or "strict", optional
        ``True`` (default): refuse files with errors, show the warnings.
        ``"strict"``: also refuse files with warnings.
        ``False``: no check, to open a damaged file and repair it.
    u : (float, float), optional
        Read only the part of the road between these two u values (m). Only
        this part is loaded in memory: use it for files larger than the memory.
    step : int or (int, int), optional
        Keep one cross section out of ``step`` (and one long section out of
        ``step``; give two values to choose them separately). A quick, light
        preview of a very large file, e.g. ``step=4`` uses 16 times less memory.
    dtype : str, optional
        Type of the elevation grid in memory: ``"float64"`` (default) or
        ``"float32"``, which uses half the memory and keeps a resolution
        far below a micrometre on road heights. The reference line is
        always kept in float64.

    Returns
    -------
    CrgRoad

    Raises
    ------
    pyopencrg.CrgCheckError
        If the file has errors (or warnings with ``check="strict"``).

    Examples
    --------
    >>> import pyopencrg as crg
    >>> road = crg.read("belgian_block.crg")    # doctest: +SKIP
    >>> print(road)                             # doctest: +SKIP

    Preview of a huge scan, then full resolution on 20 m::

    >>> overview = crg.read("scan.crg", step=4)       # doctest: +SKIP
    >>> detail = crg.read("scan.crg", u=(500, 520))   # doctest: +SKIP
    """
    from .io import CrgCorruptedError, read_crg_file
    if apply_mods and (u is not None or step != 1):  # modifiers of the whole file, then the part
        from .crgfile import CrgFile
        with CrgFile(path, check) as f:
            return f.read(u=u, step=step, dtype=dtype, apply_mods=True)
    try:
        content = read_crg_file(path, u=u, step=step, stop_on_error=bool(check), dtype=_grid_dtype(dtype))
    except CrgCorruptedError as exc:  # refused before the data is loaded in memory
        from .checks import CheckReport, CrgCheckError
        raise CrgCheckError(CheckReport(errors=[f"(reading) {e}" for e in exc.errors]), os.fspath(path)) from None
    road = _road_from_content(content)
    if check:
        from .checks import validate
        validate(road, strict=(check == "strict"))
    if apply_mods:
        road = road.apply_mods()
    return road

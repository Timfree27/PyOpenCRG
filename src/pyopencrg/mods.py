"""
Modifiers: scaling, NaN handling and re-positioning of a road.

The ``$ROAD_CRG_MODS`` block of a file lists modifiers that a simulator
applies when it loads the road. They are applied in this order:

1. scaling (``scale_z_grid``, ``scale_slope``, ``scale_banking``,
   ``scale_length``, ``scale_width``, ``scale_curvature``);
2. NaN handling (``grid_nan_mode``, ``grid_nan_offset``);
3. re-positioning "by reference point" (``refpoint_*``): a point (u, v) of the
   road is moved to (x, y, z) with heading phi;
4. re-positioning "by offset" (``refline_offset_*``, ``refline_rotcenter_*``):
   the road is rotated then translated.

The functions :func:`move_to`, :func:`transform` and :func:`scale` give a
direct access to the same operations.
"""

from __future__ import annotations

import warnings

import numpy as np

from .keywords import MODS_BY_OFFSET, MODS_BY_REFPOINT
from .road import integrate_heading

__all__ = ["apply_mods", "move_to", "transform", "scale"]


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def transform(road, dx=0.0, dy=0.0, dz=0.0, dphi=0.0, center=None):
    """Rotate a road by ``dphi`` (rad) around ``center`` (x, y), then translate it.

    The geodetic offsets (``reference_line_offset_*``) are updated so that the
    road keeps its geographic position.

    Parameters
    ----------
    road : CrgRoad
    dx, dy, dz : float
        Translation (m).
    dphi : float
        Rotation angle (rad, counter clockwise).
    center : (float, float), optional
        Rotation center. Default: start of the reference line.

    Returns
    -------
    CrgRoad
    """
    x, y = road.x, road.y
    cx, cy = (x[0], y[0]) if center is None else (float(center[0]), float(center[1]))
    c, s = np.cos(dphi), np.sin(dphi)
    xr = cx + (x - cx) * c - (y - cy) * s + dx
    yr = cy + (x - cx) * s + (y - cy) * c + dy
    new = road._derive(x=xr, y=yr, zr=road.zr + dz)
    new.x_offset = road.x_offset - (xr[0] - x[0])
    new.y_offset = road.y_offset - (yr[0] - y[0])
    new.z_offset = road.z_offset - dz
    new.phi_offset = _wrap(road.phi_offset - dphi)
    return new


def _move_params(road, x, y, z, phi, u, v, center, surface=None):
    """Arguments of :func:`transform` that bring the point (u, v) of ``road`` to (x, y, z, phi).

    ``surface``: road that gives the elevation at (u, v), if not ``road``.
    """
    u = road.u_start if u is None else float(u)
    v = float(v)
    px, py = road.uv_to_xy(u, v)
    surface = road if surface is None else surface
    pz = surface.uv_to_z(u, v)
    if np.isnan(pz):
        pz = float(np.interp(u, surface.u, surface.zr))
        warnings.warn("reference point lies on a NaN area: the reference line elevation is used", stacklevel=3)
    dphi = phi - road.u_to_phi(u)
    cx, cy = center
    c, s = np.cos(dphi), np.sin(dphi)
    qx = cx + (px - cx) * c - (py - cy) * s
    qy = cy + (px - cx) * s + (py - cy) * c
    return x - qx, y - qy, z - pz, dphi, center


def move_to(road, x=0.0, y=0.0, z=0.0, phi=0.0, u=None, v=0.0):
    """Move a road so that its point (u, v) is at (x, y, z) with heading ``phi``.

    Parameters
    ----------
    road : CrgRoad
    x, y, z : float
        Target position (m).
    phi : float
        Target heading of the reference line at the reference point (rad).
    u, v : float, optional
        Reference point (default: start of the reference line, v = 0).

    Returns
    -------
    CrgRoad

    Examples
    --------
    Put the start of the road at the origin, heading along x::

        >>> moved = road.move_to(0, 0, 0, 0)    # doctest: +SKIP
    """
    return transform(road, *_move_params(road, x, y, z, phi, u, v, (road.x[0], road.y[0])))


def scale(road, z=1.0, slope=1.0, banking=1.0, length=1.0, width=1.0, curvature=1.0):
    """Scale a road.

    Parameters
    ----------
    z : float
        Factor applied to the elevation grid.
    slope : float
        Factor applied to the longitudinal slope.
    banking : float
        Factor applied to the banking.
    length : float
        Factor applied to all lengths along u (the increment changes).
    width : float
        Factor applied to the lateral positions v.
    curvature : float
        Factor applied to the curvature of the reference line.

    Returns
    -------
    CrgRoad
    """
    new = road.copy()
    if z != 1.0:
        new._z = new._z * z
    if slope != 1.0:
        new._zr = new._zr[0] + slope * (new._zr - new._zr[0])
    if banking != 1.0:
        new._banking = new._banking * banking
    if length != 1.0:
        if length <= 0:
            raise ValueError("scale_length must be positive")
        x0, y0 = new.x[0], new.y[0]
        new = new._derive(u_inc=new.u_inc * length, x=x0 + length * (new.x - x0),
                          y=y0 + length * (new.y - y0), zr=new._zr[0] + length * (new._zr - new._zr[0]))
        new.lon_end = new.lat_end = None
    if width != 1.0:
        if width <= 0:
            raise ValueError("scale_width must be positive")
        new = new._derive(v=new.v * width)
    if curvature != 1.0 and not new.is_straight:
        p = np.unwrap(new.phi)
        p = p[0] + np.concatenate(([0.0], np.cumsum(curvature * np.diff(p))))
        x, y = integrate_heading(new.x[0], new.y[0], p, new.u_inc)
        new = new._derive(x=x, y=y)
        new.lon_end = new.lat_end = None
    if new.alt_start is not None:
        new.alt_end = new.alt_start + (new.zr[-1] - new.zr[0])
    return new


def _plan(mods):
    """Check the modifiers; returns (NaN filling (mode, offset) or None, by reference point, by offset)."""
    for key in ("scale_length", "scale_width"):
        if key in mods and mods[key] <= 0:
            raise ValueError(f"illegal modifier {key} = {mods[key]}")
    nan = None
    if "grid_nan_mode" in mods or "grid_nan_offset" in mods:
        mode = int(mods.get("grid_nan_mode", 2))
        if mode not in (0, 1, 2):
            raise ValueError(f"illegal modifier grid_nan_mode = {mode}")
        if mode == 0 and "grid_nan_offset" in mods:
            raise ValueError("grid_nan_offset cannot be used with grid_nan_mode = 0")
        if mode != 0:
            nan = (mode, mods.get("grid_nan_offset", 0.0))
    by_ref = any(k in mods for k in MODS_BY_REFPOINT) or any(k in mods for k in ("refpoint_u_offset", "refpoint_v_offset"))
    by_off = any(k in mods for k in MODS_BY_OFFSET)
    if by_ref and by_off:
        warnings.warn("re-positioning 'by reference point' overwrites the 'by offset' modifiers", stacklevel=3)
        by_off = False
    if by_ref:
        if "refpoint_u_fraction" in mods and "refpoint_u" in mods:
            raise ValueError("refpoint_u and refpoint_u_fraction cannot be both defined")
        if "refpoint_v_fraction" in mods and "refpoint_v" in mods:
            raise ValueError("refpoint_v and refpoint_v_fraction cannot be both defined")
    return nan, by_ref, by_off


def _scale(road, mods):
    return scale(road, z=mods.get("scale_z_grid", 1.0), slope=mods.get("scale_slope", 1.0),
                 banking=mods.get("scale_banking", 1.0), length=mods.get("scale_length", 1.0),
                 width=mods.get("scale_width", 1.0), curvature=mods.get("scale_curvature", 1.0))


def _refpoint(mods, road):
    """(u, v) of the reference point on the scaled road."""
    if "refpoint_u_fraction" in mods:
        u = road.u_start + mods["refpoint_u_fraction"] * (road.u_end - road.u_start)
    else:
        u = mods.get("refpoint_u", road.u_start)
    u += mods.get("refpoint_u_offset", 0.0)
    if "refpoint_v_fraction" in mods:
        v = road.v_min + mods["refpoint_v_fraction"] * (road.v_max - road.v_min)
    else:
        v = mods.get("refpoint_v", 0.0)
    v += mods.get("refpoint_v_offset", 0.0)
    return u, v


def _reposition(road, mods, by_ref, by_off, surface):
    """Steps 3 and 4 on ``road`` (scaled, with the whole reference line).

    ``surface(u)`` gives the scaled and NaN filled road around u.
    """
    if by_ref:
        u, v = _refpoint(mods, road)
        # position and heading from the whole road, elevation from the surface around the point
        road = transform(road, *_move_params(road, mods.get("refpoint_x", 0.0), mods.get("refpoint_y", 0.0),
                                             mods.get("refpoint_z", 0.0), mods.get("refpoint_phi", 0.0), u, v,
                                             (road.x[0], road.y[0]), surface(u)))
    if by_off:
        center = (mods.get("refline_rotcenter_x", road.x[0]), mods.get("refline_rotcenter_y", road.y[0]))
        road = transform(road, mods.get("refline_offset_x", 0.0), mods.get("refline_offset_y", 0.0),
                         mods.get("refline_offset_z", 0.0), mods.get("refline_offset_phi", 0.0), center)
    return road


def apply_mods(road):
    """Apply the modifiers of a road and return the modified road.

    See :meth:`pyopencrg.CrgRoad.apply_mods`.
    """
    from .edit import fill_nan

    mods = road.effective_mods()
    nan, by_ref, by_off = _plan(mods)
    if road._part and (by_ref or by_off):
        warnings.warn("this road is a part of a file: it is moved relative to the part, not to the whole file. "
                      "Read it with apply_mods=True (crg.read(path, u=..., apply_mods=True) or "
                      "f.read(u=..., apply_mods=True)) to place it like in the whole road.", stacklevel=3)
    new = _scale(road, mods)
    if nan is not None:
        new = fill_nan(new, *nan)
    new = _reposition(new, mods, by_ref, by_off, lambda u: new)
    new.mods = {}
    return new


def _file_mods(geo, v, read_rows):
    """Modifiers of a whole file, computed once, for the parts read with ``apply_mods=True``.

    ``geo`` holds the reference line of all the cross sections of the file
    (its grid is not used), ``v`` the lateral positions of the file and
    ``read_rows(i0, i1)`` returns the rows ``i0:i1`` of the grid as stored in
    the file. Only the few rows around the reference point are read.

    Returns (moved reference line road, z factor, v factor, NaN filling or None).
    """
    from .edit import fill_nan
    from .road import CrgRoad, _set_part_line

    mods = geo.effective_mods()
    nan, by_ref, by_off = _plan(mods)
    zs, ws = mods.get("scale_z_grid", 1.0), mods.get("scale_width", 1.0)
    road = _scale(geo, mods)

    def surface(u):
        nu = road.nu
        i = (u - road.u_start) / road.u_inc
        outside = not 0 <= i <= nu - 1
        if outside and (road.get_option("border_mode_u") in (3, 4) or road.get_option("refline_continuation")):
            i0, i1 = 0, nu  # the value depends on the other end of the road
        else:  # enough rows that the smoothing zones of the window ends do not reach the point
            n = 3 + int(np.ceil(max(road.get_option("border_smooth_ubeg"), road.get_option("border_smooth_uend"))
                                / road.u_inc))
            c = int(np.clip(np.floor(i), 0, nu - 1))
            i0, i1 = max(0, c - n), min(nu, c + n + 2)
        z = np.asarray(read_rows(i0, i1), dtype=np.float64) * zs
        part = CrgRoad._from_state(z, v * ws, road.u_start + i0 * road.u_inc, road.u_inc, road.x[i0:i1],
                                   road.y[i0:i1], road.zr[i0:i1], road.banking[i0:i1], template=road)
        part = part if nan is None else fill_nan(part, *nan)
        _set_part_line(part, road.x, road.y, np.arange(i0, i1), road.is_straight)
        return part

    road = _reposition(road, mods, by_ref, by_off, surface)
    road.mods = {}
    return road, zs, ws, nan

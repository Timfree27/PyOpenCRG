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
    u = road.u_start if u is None else float(u)
    v = float(v)
    px, py = road.uv_to_xy(u, v)
    pz = road.uv_to_z(u, v)
    if np.isnan(pz):
        pz = float(np.interp(u, road.u, road.zr))
        warnings.warn("reference point lies on a NaN area: the reference line elevation is used", stacklevel=2)
    heading = road.u_to_phi(u)
    dphi = phi - heading
    x0, y0 = road.x[0], road.y[0]
    c, s = np.cos(dphi), np.sin(dphi)
    qx = x0 + (px - x0) * c - (py - y0) * s
    qy = y0 + (px - x0) * s + (py - y0) * c
    return transform(road, x - qx, y - qy, z - pz, dphi, (x0, y0))


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


def apply_mods(road):
    """Apply the modifiers of a road and return the modified road.

    See :meth:`pyopencrg.CrgRoad.apply_mods`.
    """
    from .edit import fill_nan

    mods = road.effective_mods()
    new = road.copy()

    # 1. scaling
    for key in ("scale_length", "scale_width"):
        if key in mods and mods[key] <= 0:
            raise ValueError(f"illegal modifier {key} = {mods[key]}")
    new = scale(new, z=mods.get("scale_z_grid", 1.0), slope=mods.get("scale_slope", 1.0),
                banking=mods.get("scale_banking", 1.0), length=mods.get("scale_length", 1.0),
                width=mods.get("scale_width", 1.0), curvature=mods.get("scale_curvature", 1.0))

    # 2. NaN handling
    if "grid_nan_mode" in mods or "grid_nan_offset" in mods:
        mode = int(mods.get("grid_nan_mode", 2))
        if mode not in (0, 1, 2):
            raise ValueError(f"illegal modifier grid_nan_mode = {mode}")
        if mode == 0 and "grid_nan_offset" in mods:
            raise ValueError("grid_nan_offset cannot be used with grid_nan_mode = 0")
        if mode != 0:
            new = fill_nan(new, mode, mods.get("grid_nan_offset", 0.0))

    by_ref = any(k in mods for k in MODS_BY_REFPOINT) or any(k in mods for k in ("refpoint_u_offset", "refpoint_v_offset"))
    by_off = any(k in mods for k in MODS_BY_OFFSET)
    if by_ref and by_off:
        warnings.warn("re-positioning 'by reference point' overwrites the 'by offset' modifiers", stacklevel=2)
        by_off = False

    # 3. by reference point
    if by_ref:
        if "refpoint_u_fraction" in mods and "refpoint_u" in mods:
            raise ValueError("refpoint_u and refpoint_u_fraction cannot be both defined")
        if "refpoint_v_fraction" in mods and "refpoint_v" in mods:
            raise ValueError("refpoint_v and refpoint_v_fraction cannot be both defined")
        if "refpoint_u_fraction" in mods:
            u = new.u_start + mods["refpoint_u_fraction"] * (new.u_end - new.u_start)
        else:
            u = mods.get("refpoint_u", new.u_start)
        u += mods.get("refpoint_u_offset", 0.0)
        if "refpoint_v_fraction" in mods:
            v = new.v_min + mods["refpoint_v_fraction"] * (new.v_max - new.v_min)
        else:
            v = mods.get("refpoint_v", 0.0)
        v += mods.get("refpoint_v_offset", 0.0)
        new = move_to(new, mods.get("refpoint_x", 0.0), mods.get("refpoint_y", 0.0),
                      mods.get("refpoint_z", 0.0), mods.get("refpoint_phi", 0.0), u, v)

    # 4. by offset
    if by_off:
        center = (mods.get("refline_rotcenter_x", new.x[0]), mods.get("refline_rotcenter_y", new.y[0]))
        new = transform(new, mods.get("refline_offset_x", 0.0), mods.get("refline_offset_y", 0.0),
                        mods.get("refline_offset_z", 0.0), mods.get("refline_offset_phi", 0.0), center)

    new.mods = {}
    return new

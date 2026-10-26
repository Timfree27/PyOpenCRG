"""
Interactive 3D viewer based on PyVista (VTK, rendered by the graphics card).

Much faster than matplotlib for large roads: several million grid points
can be rotated and zoomed smoothly. PyVista is an optional dependency::

    pip install "pyopencrg[viewer]"

Usage::

    road.view3d()                          # absolute surface coloured by height
    road.view3d(color="grid", z_scale=5)   # colour = grid height (texture), z exaggerated 5 times
    road.view3d(color="normal")            # colour = normal map (shows scan defects)
    road.view3d(screenshot="road.png", off_screen=True)   # picture without window

Mouse: left button = rotate, right button / wheel = zoom, middle button = pan.
Keys: ``r`` = reset the view, ``q`` = quit.
"""

from __future__ import annotations

import numpy as np

__all__ = ["view3d"]

_COLORS = {
    "z": "absolute height (m)",
    "grid": "grid height (m)",
    "normal": "normal vertical component",
}


def _pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise ImportError("the 3D viewer needs PyVista: pip install \"pyopencrg[viewer]\" "
                          "(or use road.show() for matplotlib figures)") from exc
    return pv


def _indices(n, step):
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def view3d(road, color: str = "z", absolute: bool = True, z_scale: float = 1.0, step=None,
           max_points: int = 4_000_000, u=None, cmap=None, show_reference_line: bool = True,
           off_screen: bool = False, screenshot=None, show: bool = True):
    """Show a road in an interactive 3D window.

    Parameters
    ----------
    road : CrgRoad
    color : str
        What the colours show: ``"z"`` (absolute height), ``"grid"`` (grid
        height, i.e. the texture without slope and banking) or ``"normal"``
        (normal map, to spot scanning defects).
    absolute : bool
        Draw the absolute surface (with slope and banking). If False, only
        the grid is drawn, flat along the reference line.
    z_scale : float
        Vertical exaggeration (the axis labels stay in true metres).
    step : int, optional
        Keep one grid point out of ``step``. By default it is chosen so that
        at most ``max_points`` points are drawn.
    max_points : int
        Maximum number of grid points drawn when ``step`` is not given.
    u : (float, float), optional
        Show only this part of the road (m).
    cmap : str, optional
        Matplotlib colour map name (default: ``"viridis"``, ``"Blues_r"`` for normals).
    show_reference_line : bool
        Draw the reference line in red.
    off_screen : bool
        Render without opening a window (for screenshots on a server).
    screenshot : str, optional
        Save a picture of the view to this file.
    show : bool
        Open the window (or render, when ``off_screen``). If False, the
        plotter is returned without being shown, to add your own objects.

    Returns
    -------
    pyvista.Plotter
    """
    pv = _pyvista()
    if color not in _COLORS:
        raise ValueError(f"color must be one of {list(_COLORS)}")
    sub = road if u is None else road.crop(u=u)
    if step is None:
        step = max(1, int(np.ceil(np.sqrt(sub.nu * sub.nv / max_points))))
    iu, iv = _indices(sub.nu, step), _indices(sub.nv, step)
    uu, vv = np.meshgrid(sub.u[iu], sub.v[iv], indexing="ij")
    x, y = sub.uv_to_xy(uu, vv)
    # only the drawn points are computed (no full size copy of a huge grid)
    grid_z = sub.z[np.ix_(iu, iv)]
    z = grid_z + sub.zr[iu, None] + sub.banking[iu, None] * sub.v[None, iv] if absolute else grid_z.copy()
    if color == "z":
        values = z.copy()
    elif color == "grid":
        values = grid_z.copy()
    else:
        gu, gv = np.gradient(grid_z, sub.u[iu], sub.v[iv])
        values = 1.0 / np.sqrt(1.0 + gu ** 2 + gv ** 2)
    nan = ~np.isfinite(z)
    if nan.any():  # VTK needs finite coordinates: NaN points are drawn transparent
        z[nan] = np.nanmean(z) if (~nan).any() else 0.0
        values[nan] = np.nan
    mesh = pv.StructuredGrid(x, y, z)
    mesh.point_data[_COLORS[color]] = values.ravel(order="F")

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(mesh, scalars=_COLORS[color], cmap=cmap or ("Blues_r" if color == "normal" else "viridis"),
                     nan_opacity=0.0, show_scalar_bar=True, smooth_shading=True)
    if show_reference_line:
        k = _indices(sub.nu, max(1, sub.nu // 20000))
        zl = sub.uv_to_z(sub.u[k], np.zeros(k.size), grid_only=not absolute)  # drawn on the surface
        zl = np.where(np.isfinite(zl), zl, np.nanmean(z))
        line = pv.lines_from_points(np.column_stack((sub.x[k], sub.y[k], zl + 1e-3)))
        plotter.add_mesh(line, color="red", line_width=3)
    plotter.set_scale(zscale=z_scale)
    plotter.add_axes()
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    plotter.show_grid(xtitle="x (m)", ytitle="y (m)", ztitle="z (m)", n_zlabels=3, font_size=9, fmt="%.2f",
                      axes_ranges=[float(x.min()), float(x.max()), float(y.min()), float(y.max()), zmin, zmax])
    plotter.add_text(f"{road!r}"[1:-1], font_size=9)
    plotter.view_isometric()
    if show:
        plotter.show(screenshot=screenshot)
    elif screenshot:
        plotter.screenshot(screenshot)
    return plotter

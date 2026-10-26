"""
Plots of a road (matplotlib).

The quickest way is ``road.show()``, which opens the standard set of
overview figures. Every plot is also available on its own, either as a
function of this module or through ``road.plot``::

    road.plot.refline_map()            # same as pyopencrg.plot.refline_map(road)
    road.plot.heightmap()
    road.plot.cross_sections()

Functions drawing in a single axes accept an ``ax`` argument to draw in an
existing matplotlib axes; they return the axes. Functions starting with
``show_`` create a whole figure and return it.

Large grids are automatically decimated so that plots stay fast.
"""

from __future__ import annotations

import functools
import os
from datetime import datetime

import numpy as np

from ._optional import need

__all__ = [
    "show", "show_refline_map", "show_refline_elevation", "show_cuts_and_limits", "show_elgrid_surface",
    "show_road_surface", "show_road_uv_surface", "show_info", "show_refpoints",
    "refline_xy", "heading", "refline_xy_curvature", "curvature", "refline_elevation", "slope_banking",
    "refline_xyz", "grid_limits", "cross_sections", "long_sections", "elgrid_uvz_map", "elgrid_xyz_map",
    "road_uvz_map", "road_xyz_map", "heightmap", "normalmap", "normal_distribution", "peaks", "comparison",
    "refpoint_distances", "PlotAccessor",
]

# categorical colors, used in this fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK = "#3b3b38"
GRID = "#d9d8d2"
MAX_LINE_POINTS = 20000
MAX_MAP = (2000, 600)
MAX_SURF = (250, 120)


def _plt():
    plt = need("matplotlib.pyplot", "the plots")
    return plt


def _style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    if ax.name != "3d":
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return ax


def _ax(ax, projection=None):
    if ax is not None:
        return ax
    plt = _plt()
    fig = plt.figure(figsize=(8, 5))
    return fig.add_subplot(111, projection=projection)


def _figure(road, title, nrows=2, ncols=2, figsize=(12, 8)):
    plt = _plt()
    fig = plt.figure(figsize=figsize)
    name = os.path.basename(road.filename) if getattr(road, "filename", "") else "unsaved road"
    fig.suptitle(title, fontsize=13, color=INK)
    fig.text(0.01, 0.005, name, fontsize=7, color="#7a7970")
    fig.text(0.99, 0.005, datetime.now().strftime("%Y-%m-%d %H:%M"), fontsize=7, color="#7a7970", ha="right")
    return fig


def _range(road, iu, iv):
    from .edit import _index_range
    return _index_range(iu, road.nu, "u"), _index_range(iv, road.nv, "v")


def _idx(i0, i1, nmax):
    step = max(1, int(np.ceil((i1 - i0 + 1) / nmax)))
    idx = np.arange(i0, i1 + 1, step)
    if idx[-1] != i1:
        idx = np.append(idx, i1)
    return idx


def _marks(ax, x, y):
    ax.plot(x[0], y[0], ">", color=INK, markersize=8, label="start")
    ax.plot(x[-1], y[-1], "s", color=INK, markersize=8, label="end")


# ---------------------------------------------------------------------------
# reference line
# ---------------------------------------------------------------------------

def refline_xy(road, iu=None, ax=None):
    """Reference line in the x/y plane (start ``>``, end ``□``)."""
    ax = _ax(ax)
    (i0, i1), _ = _range(road, iu, None)
    k = _idx(i0, i1, MAX_LINE_POINTS)
    ax.plot(road.x[k], road.y[k], color=SERIES[0], linewidth=2)
    _marks(ax, road.x[k], road.y[k])
    ax.set_aspect("equal", adjustable="datalim")
    return _style(ax, "x (m)", "y (m)", "reference line")


def heading(road, iu=None, ax=None):
    """Heading angle of the reference line along u (degrees)."""
    ax = _ax(ax)
    (i0, i1), _ = _range(road, iu, None)
    k = _idx(i0, min(i1, road.nu - 2), MAX_LINE_POINTS)
    phi = np.degrees(np.unwrap(road.phi))
    ax.plot(road.u[k], phi[k], color=SERIES[0], linewidth=2)
    return _style(ax, "u (m)", "heading (deg)", "heading angle")


def curvature(road, iu=None, ax=None):
    """Curvature of the reference line along u (1/m, positive = left)."""
    ax = _ax(ax)
    (i0, i1), _ = _range(road, iu, None)
    rc = np.concatenate(([0.0], road.curvature, [0.0])) if road.nu > 2 else np.zeros(road.nu)
    k = _idx(i0, i1, MAX_LINE_POINTS)
    ax.plot(road.u[k], rc[k], color=SERIES[0], linewidth=2)
    ax.axhline(0, color=INK, linewidth=0.8)
    return _style(ax, "u (m)", "curvature (1/m)", "curvature")


def refline_xy_curvature(road, iu=None, ax=None):
    """Reference line with its curvature drawn as an offset curve."""
    ax = _ax(ax)
    (i0, i1), _ = _range(road, iu, None)
    k = _idx(i0, i1, MAX_LINE_POINTS)
    x, y = road.x[k], road.y[k]
    rc = (np.concatenate(([0.0], road.curvature, [0.0])) if road.nu > 2 else np.zeros(road.nu))[k]
    ph = np.concatenate((road.phi, [road.phi[-1]]))[k]
    rcmax = np.max(np.abs(rc)) + road.get_option("check_eps")
    scale = -0.1 * (i1 - i0 + 1) * road.u_inc / rcmax
    ax.plot(x, y, color=SERIES[0], linewidth=2, label="reference line")
    ax.plot(x - scale * rc * np.sin(ph), y + scale * rc * np.cos(ph), color=SERIES[1], linewidth=1.5,
            label="curvature (scaled)")
    _marks(ax, x, y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return _style(ax, "x (m)", "y (m)", "reference line and curvature")


def refline_elevation(road, iu=None, ax=None):
    """Elevation of the reference line along u."""
    ax = _ax(ax)
    (i0, i1), _ = _range(road, iu, None)
    k = _idx(i0, i1, MAX_LINE_POINTS)
    ax.plot(road.u[k], road.zr[k], color=SERIES[0], linewidth=2)
    return _style(ax, "u (m)", "z (m)", "reference line elevation")


def slope_banking(road, iu=None, ax=None):
    """Slope and banking of the reference line along u (percent)."""
    ax = _ax(ax)
    (i0, i1), _ = _range(road, iu, None)
    k = _idx(i0, i1, MAX_LINE_POINTS)
    s = np.concatenate((road.slope, [road.slope[-1]]))
    ax.plot(road.u[k], 100 * s[k], color=SERIES[0], linewidth=2, label="slope")
    ax.plot(road.u[k], 100 * road.banking[k], color=SERIES[1], linewidth=2, label="banking")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return _style(ax, "u (m)", "%", "slope and banking")


def refline_xyz(road, iu=None, ax=None):
    """Reference line in 3D."""
    ax = _ax(ax, "3d")
    (i0, i1), _ = _range(road, iu, None)
    k = _idx(i0, i1, MAX_LINE_POINTS)
    ax.plot(road.x[k], road.y[k], road.zr[k], color=SERIES[0], linewidth=2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    return _style(ax, title="reference line (3D)")


# ---------------------------------------------------------------------------
# elevation grid
# ---------------------------------------------------------------------------

def grid_limits(road, iu=None, iv=None, ax=None):
    """Right and left limits of the valid (non NaN) data along u."""
    ax = _ax(ax)
    (i0, i1), (j0, j1) = _range(road, iu, iv)
    k = _idx(i0, i1, MAX_LINE_POINTS)
    valid = np.isfinite(road.z[k][:, j0:j1 + 1])
    has = valid.any(axis=1)
    first = np.argmax(valid, axis=1) + j0
    last = j1 - np.argmax(valid[:, ::-1], axis=1)
    vl = np.where(has, road.v[last], np.nan)
    vr = np.where(has, road.v[first], np.nan)
    ax.plot(road.u[k], vl, color=SERIES[0], linewidth=2, label="left limit")
    ax.plot(road.u[k], vr, color=SERIES[1], linewidth=2, label="right limit")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return _style(ax, "u (m)", "v (m)", "valid data limits")


def _seq_colors(n):
    colormaps = need("matplotlib", "the plots").colormaps
    return colormaps["Blues"](np.linspace(0.35, 0.95, n))


def cross_sections(road, iu=None, iv=None, n=10, absolute=False, ax=None):
    """About ``n`` cross sections (z over v), evenly spread along u."""
    ax = _ax(ax)
    (i0, i1), (j0, j1) = _range(road, iu, iv)
    rows = np.unique(np.linspace(i0, i1, min(n, i1 - i0 + 1)).astype(int))
    z = road.surface_z() if absolute else road.z
    for c, i in zip(_seq_colors(rows.size), rows):
        ax.plot(road.v[j0:j1 + 1], z[i, j0:j1 + 1], color=c, linewidth=1.5, label=f"u = {road.u[i]:.2f} m")
    ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    return _style(ax, "v (m)", "z (m)", "cross sections (light = start, dark = end)")


def long_sections(road, iu=None, iv=None, n=10, absolute=False, ax=None):
    """About ``n`` long sections (z over u), evenly spread along v."""
    ax = _ax(ax)
    (i0, i1), (j0, j1) = _range(road, iu, iv)
    cols = np.unique(np.linspace(j0, j1, min(n, j1 - j0 + 1)).astype(int))
    k = _idx(i0, i1, MAX_LINE_POINTS)
    z = road.surface_z() if absolute else road.z
    for c, j in zip(_seq_colors(cols.size), cols):
        ax.plot(road.u[k], z[k, j], color=c, linewidth=1, label=f"v = {road.v[j]:.2f} m")
    ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    return _style(ax, "u (m)", "z (m)", "long sections (light = right, dark = left)")


class ZoomableImage:
    """Image of a (u, v) grid that is re-sampled when you zoom.

    The grid is drawn with ``imshow``, decimated to about the number of
    screen pixels of the axes. When the view changes (zoom, pan), the
    visible part is extracted again at the best resolution, so that zooming
    in shows every grid point. The colour scale stays fixed.

    The object is kept in ``ax.zoomable_image``.
    """

    def __init__(self, ax, u, v, data, i_range, j_range, cmap, label):
        self.ax, self.u, self.v, self.data = ax, u, v, data
        self.i0, self.i1 = i_range
        self.j0, self.j1 = j_range
        self.du = u[1] - u[0]
        self.dv = v[1] - v[0]
        sub = data[self.i0:self.i1 + 1:max(1, (self.i1 - self.i0) // 2000 + 1),
                   self.j0:self.j1 + 1:max(1, (self.j1 - self.j0) // 1000 + 1)]
        fin = sub[np.isfinite(sub)]
        vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
        self.im = ax.imshow(np.zeros((2, 2)), origin="lower", aspect="auto", cmap=cmap,
                            interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.figure.colorbar(self.im, ax=ax, label=label)
        self._busy = False
        self.update(self.i0, self.i1, self.j0, self.j1)
        ax.set_xlim(u[self.i0] - self.du / 2, u[self.i1] + self.du / 2)
        ax.set_ylim(v[self.j0] - self.dv / 2, v[self.j1] + self.dv / 2)
        ax.set_autoscale_on(False)
        ax.callbacks.connect("xlim_changed", self._on_change)
        ax.callbacks.connect("ylim_changed", self._on_change)
        ax.zoomable_image = self  # keep a reference (callbacks are weak references)

    def _pixels(self):
        bbox = self.ax.get_window_extent()
        return max(200, int(bbox.width)), max(100, int(bbox.height))

    def update(self, ia, ib, ja, jb):
        """Draw the grid part ``[ia, ib] x [ja, jb]`` (indices) at screen resolution."""
        pw, ph = self._pixels()
        su = max(1, int(np.ceil((ib - ia + 1) / pw)))
        sv = max(1, int(np.ceil((jb - ja + 1) / ph)))
        sub = self.data[ia:ib + 1:su, ja:jb + 1:sv]
        nu_, nv_ = sub.shape
        self._busy = True
        try:
            self.im.set_data(np.ma.masked_invalid(sub.T))
            self.im.set_extent((self.u[ia] - su * self.du / 2, self.u[ia] + (nu_ - 0.5) * su * self.du,
                                self.v[ja] - sv * self.dv / 2, self.v[ja] + (nv_ - 0.5) * sv * self.dv))
        finally:
            self._busy = False
        self.step = (su, sv)

    def _on_change(self, ax):
        if self._busy:
            return
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        ia = int(np.clip(np.floor((x0 - self.u[0]) / self.du), self.i0, self.i1))
        ib = int(np.clip(np.ceil((x1 - self.u[0]) / self.du), self.i0, self.i1))
        ja = int(np.clip(np.floor((y0 - self.v[0]) / self.dv), self.j0, self.j1))
        jb = int(np.clip(np.ceil((y1 - self.v[0]) / self.dv), self.j0, self.j1))
        self.update(ia, max(ib, ia + 1), ja, max(jb, ja + 1))
        ax.figure.canvas.draw_idle()


def _regular(a):
    d = np.diff(a)
    return d.size > 0 and np.ptp(d) <= 1e-9 * max(abs(a[-1] - a[0]), 1.0)


def _grid_image(ax, u, v, data, i_range, j_range, cmap, label):
    """imshow with zoom re-sampling for a regular grid, pcolormesh otherwise."""
    if _regular(u) and _regular(v):
        ZoomableImage(ax, u, v, data, i_range, j_range, cmap, label)
        return
    (i0, i1), (j0, j1) = i_range, j_range
    k = _idx(i0, i1, MAX_MAP[0])
    m = _idx(j0, j1, MAX_MAP[1])
    im = ax.pcolormesh(u[k], v[m], np.ma.masked_invalid(data[np.ix_(k, m)].T), cmap=cmap, shading="auto")
    ax.figure.colorbar(im, ax=ax, label=label)


def _uv_image(road, data, iu, iv, ax, title, cmap, label):
    ax = _ax(ax)
    i_range, j_range = _range(road, iu, iv)
    _grid_image(ax, road.u, road.v, data, i_range, j_range, cmap, label)
    return _style(ax, "u (m)", "v (m)", title)


def elgrid_uvz_map(road, iu=None, iv=None, ax=None, cmap="viridis"):
    """Elevation grid as a flat (u, v) image."""
    return _uv_image(road, road.z, iu, iv, ax, "elevation grid (u, v)", cmap, "z grid (m)")


def road_uvz_map(road, iu=None, iv=None, ax=None, cmap="viridis"):
    """Absolute road surface as a flat (u, v) image."""
    return _uv_image(road, road.surface_z(), iu, iv, ax, "road surface (u, v)", cmap, "z (m)")


def heightmap(road, absolute=False, iu=None, iv=None, ax=None, cmap="Greys"):
    """Height map of the road (flat u, v image). Zoom in to see every grid point."""
    data = road.surface_z() if absolute else road.z
    return _uv_image(road, data, iu, iv, ax, "height map", cmap, "z (m)")


def normalmap(road, iu=None, iv=None, ax=None, cmap="Blues"):
    """Normal map: vertical component of the surface normal (1 = flat). Zoom in to see every grid point."""
    return _uv_image(road, road.normal_map(), iu, iv, ax, "normal map", cmap, "normal vertical component")


def normal_distribution(road, ax=None, bins=100):
    """Histogram of the normal map, useful to choose ``normal_tol`` of :meth:`remove_artifacts`."""
    ax = _ax(ax)
    n = road.normal_map().ravel()
    ax.hist(n[np.isfinite(n)], bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    return _style(ax, "normal vertical component", "number of grid points", "normal map distribution")


def _xyz_surface(road, data, iu, iv, ax, title, cmap):
    ax = _ax(ax, "3d")
    (i0, i1), (j0, j1) = _range(road, iu, iv)
    k = _idx(i0, i1, MAX_SURF[0])
    m = _idx(j0, j1, MAX_SURF[1])
    uu, vv = np.meshgrid(road.u[k], road.v[m], indexing="ij")
    x, y = road.uv_to_xy(uu, vv)
    z = data[np.ix_(k, m)]
    surf = ax.plot_surface(x, y, np.ma.masked_invalid(z), cmap=cmap, linewidth=0, antialiased=False)
    ax.figure.colorbar(surf, ax=ax, shrink=0.6, label="z (m)")
    span = max(np.ptp(x), np.ptp(y), 1e-9)
    zr = np.nanmax(z) - np.nanmin(z) if np.isfinite(z).any() else 1.0
    ax.set_box_aspect((np.ptp(x) / span + 1e-3, np.ptp(y) / span + 1e-3, max(0.3, min(1.0, 10 * zr / span))))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    return _style(ax, title=title)


def elgrid_xyz_map(road, iu=None, iv=None, ax=None, cmap="viridis"):
    """Elevation grid drawn in 3D at its x/y position (z exaggerated)."""
    return _xyz_surface(road, road.z, iu, iv, ax, "elevation grid (3D)", cmap)


def road_xyz_map(road, iu=None, iv=None, ax=None, cmap="viridis"):
    """Absolute road surface in 3D."""
    return _xyz_surface(road, road.surface_z(), iu, iv, ax, "road surface (3D)", cmap)


def peaks(road, result, ax=None):
    """Elevation grid with the peaks found by :meth:`pyopencrg.CrgRoad.find_peaks`."""
    ax = elgrid_uvz_map(road, ax=ax)
    ax.plot(result.u, result.v, "o", markersize=4, markerfacecolor="none", markeredgecolor=SERIES[1],
            label=f"{len(result)} peak points")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("peaks", fontsize=10, color=INK)
    return ax


def comparison(result, block=True):
    """Figure showing the result of :func:`pyopencrg.compare`."""
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    _grid_image(axs[0], result.u, result.v, result.diff, (0, result.u.size - 1), (0, result.v.size - 1),
                "Blues", "absolute difference (m)")
    _style(axs[0], "u (m)", "v (m)", "absolute difference")
    fin = result.diff[np.isfinite(result.diff)]
    axs[1].hist(fin.ravel(), bins=100, color=SERIES[0])
    axs[1].set_yscale("log")
    _style(axs[1], "absolute difference (m)", "number of points", "distribution")
    fig.suptitle(f"comparison: identical = {result.identical}, max = {result.max_diff:.3g} m", color=INK)
    if block:
        plt.show()
    return fig


def refpoint_distances(road, x, y, ax=None):
    """Lateral distance v of points (x, y) to the reference line, over u."""
    ax = _ax(ax)
    u, v = road.xy_to_uv(np.asarray(x, float), np.asarray(y, float))
    ax.plot(u, v, "o", color=SERIES[0], markersize=8)
    ax.axhline(0, color=INK, linewidth=0.8)
    return _style(ax, "u (m)", "v (m)", "distance of the points to the reference line")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def show_refline_map(road, iu=None):
    """Figure: reference line, heading, curvature."""
    fig = _figure(road, "reference line map")
    a1 = refline_xy(road, iu, fig.add_subplot(221))
    a2 = heading(road, iu, fig.add_subplot(222))
    del a1
    refline_xy_curvature(road, iu, fig.add_subplot(223))
    curvature(road, iu, fig.add_subplot(224, sharex=a2))
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def show_refline_elevation(road, iu=None):
    """Figure: reference line in 3D, elevation, slope and banking."""
    fig = _figure(road, "reference line elevation")
    refline_xyz(road, iu, fig.add_subplot(221, projection="3d"))
    refline_xy(road, iu, fig.add_subplot(222))
    a3 = refline_elevation(road, iu, fig.add_subplot(223))
    slope_banking(road, iu, fig.add_subplot(224, sharex=a3))
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def show_cuts_and_limits(road, iu=None, iv=None):
    """Figure: overview, valid data limits, cross and long sections."""
    fig = _figure(road, "elevation grid cuts and limits")
    refline_xy(road, iu, fig.add_subplot(221))
    a2 = grid_limits(road, iu, iv, fig.add_subplot(222))
    cross_sections(road, iu, iv, ax=fig.add_subplot(223))
    long_sections(road, iu, iv, ax=fig.add_subplot(224, sharex=a2))
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def show_elgrid_surface(road, iu=None, iv=None):
    """Figure: elevation grid as (u, v) image and in 3D."""
    fig = _figure(road, "elevation grid", figsize=(15, 6))
    elgrid_uvz_map(road, iu, iv, fig.add_subplot(121))
    elgrid_xyz_map(road, iu, iv, fig.add_subplot(122, projection="3d"))
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.15)
    return fig


def show_road_surface(road, iu=None, iv=None):
    """Figure: absolute road surface as (u, v) image and in 3D."""
    fig = _figure(road, "road surface", figsize=(15, 6))
    road_uvz_map(road, iu, iv, fig.add_subplot(121))
    road_xyz_map(road, iu, iv, fig.add_subplot(122, projection="3d"))
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.15)
    return fig


def show_road_uv_surface(road, u, v):
    """Figure: road surface evaluated on a grid of your choice.

    Parameters
    ----------
    u, v : array_like
        Positions along u and v where the road is evaluated.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    z = road.uv_to_z(uu, vv)
    fig =_figure(road, "road surface on a given grid", figsize=(12, 9))
    ax = fig.add_subplot(211)
    im = ax.pcolormesh(u, v, np.ma.masked_invalid(z.T), cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="z (m)")
    _style(ax, "u (m)", "v (m)", "road surface (u, v)")
    ax3 = fig.add_subplot(212, projection="3d")
    x, y = road.uv_to_xy(uu, vv)
    ax3.plot_surface(x, y, np.ma.masked_invalid(z), cmap="viridis", linewidth=0)
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def show_info(road):
    """Figure with the text summary of the road and its comments."""
    fig = _figure(road, "road information", figsize=(12, 8))
    text = road.info()
    fig.text(0.02, 0.93, text, family="monospace", fontsize=8, va="top", color=INK)
    return fig


def show_refpoints(road, x, y):
    """Figure: reference line with points, and distance of the points to it."""
    fig = _figure(road, "reference points and reference line", 1, 2, figsize=(12, 5))
    ax = refline_xy(road, ax=fig.add_subplot(121))
    ax.plot(x, y, "o", color=SERIES[1], markersize=8, label="points")
    ax.legend(loc="best", fontsize=8, frameon=False)
    refpoint_distances(road, x, y, fig.add_subplot(122))
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return fig


def show(road, iu=None, iv=None, block=True):
    """Show the standard overview figures of a road (like ``crg_show`` of OpenCRG).

    Long roads (more than 3000 cross sections) are shown at their start,
    middle and end.

    Returns
    -------
    list of matplotlib figures
    """
    (i0, i1), (j0, j1) = _range(road, iu, iv)
    iu, iv = (i0, i1), (j0, j1)
    figs = []
    if not road.is_straight:
        figs.append(show_refline_map(road, iu))
    has_sb = np.ptp(road.zr) > 0 or np.any(road.banking != 0)
    if has_sb:
        figs.append(show_refline_elevation(road, iu))
    figs.append(show_cuts_and_limits(road, iu, iv))
    n = i1 - i0 + 1
    if n > 3000:
        mid = i0 + n // 2
        parts = [(i0, i0 + 1000), (mid - 500, mid + 500), (i1 - 1000, i1)]
    else:
        parts = [iu]
    for p in parts:
        if has_sb:
            figs.append(show_elgrid_surface(road, p, iv))
        figs.append(show_road_surface(road, p, iv))
    figs.append(show_info(road))
    if block:
        _plt().show()
    return figs


class PlotAccessor:
    """``road.plot.<name>(...)`` calls ``pyopencrg.plot.<name>(road, ...)``."""

    _names = [n for n in __all__ if n not in ("PlotAccessor", "comparison")]

    def __init__(self, road):
        self._road = road

    def __getattr__(self, name):
        if name in self._names:
            return functools.partial(globals()[name], self._road)
        raise AttributeError(f"no plot named '{name}'; available: {', '.join(self._names)}")

    def __dir__(self):
        return self._names

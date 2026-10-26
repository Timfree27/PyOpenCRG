"""
Interactive cleaning of scanned roads.

:func:`clean_interactive` finds the scanning artifacts of a road, then shows
them one by one in a window with buttons:

1. "Correct" / "Skip": decide whether the defect has to be repaired;
2. the repaired patch is shown next to the original; "Validate" keeps it,
   "Cancel" leaves the original values.

Example::

    import pyopencrg as crg
    from pyopencrg.cleaner import clean_interactive

    road = crg.read("scan.crg")
    cleaned = clean_interactive(road, normal_tol=0.3, radius=15)
    cleaned.save("scan_clean.crg")
"""

from __future__ import annotations

import numpy as np

from .analysis import _fill_patch, artifact_mask, nan_regions, normal_map

__all__ = ["clean_interactive"]


def _ask(fig, labels):
    """Add two buttons to a figure and wait for a click. Returns the clicked label."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    choice = {"value": labels[1]}
    buttons = []
    for i, lab in enumerate(labels):
        ax = fig.add_axes([0.66 + 0.16 * i, 0.02, 0.15, 0.05])
        b = Button(ax, lab)

        def cb(_event, lab=lab):
            choice["value"] = lab
            plt.close(fig)

        b.on_clicked(cb)
        buttons.append(b)
    plt.show()
    return choice["value"]


def clean_interactive(road, normal_tol=0.3, radius=15, margin=50, method="rbf"):
    """Review and repair the scanning artifacts of a road one by one.

    Parameters
    ----------
    road : CrgRoad
    normal_tol, radius :
        Artifact detection, see :func:`pyopencrg.analysis.artifact_mask`.
    margin : int
        Grid points shown (and used for the repair) around each defect.
    method : str
        Interpolation used for the repair (``"rbf"``, ``"linear"``, ``"cubic"``).

    Returns
    -------
    CrgRoad
        The cleaned road (the input road is not modified).
    """
    import matplotlib.pyplot as plt

    mask = artifact_mask(road, normal_tol, radius)
    holes = road.z.copy()
    holes[mask] = np.nan
    z = road.z.copy()
    normals = normal_map(road)
    du = road.u_inc
    dv = road.v_inc if road.v_inc is not None else float(np.mean(np.diff(road.v)))
    regions = nan_regions(holes, margin)
    print(f"{len(regions)} defect(s) to review")
    for n, (i0, i1, j0, j1) in enumerate(regions, 1):
        sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        uu, vv = np.meshgrid(road.u[sl[0]], road.v[sl[1]], indexing="ij")
        fig = plt.figure(figsize=(13, 6))
        fig.suptitle(f"defect {n} / {len(regions)}  (u = {road.u[i0]:.2f} .. {road.u[i1]:.2f} m)")
        ax = fig.add_subplot(131)
        ax.imshow(normals[sl].T, origin="lower", cmap="Blues", aspect="auto")
        ax.set_title("normal map")
        ax = fig.add_subplot(132)
        ax.imshow(np.isnan(holes[sl]).T, origin="lower", cmap="gray", aspect="auto")
        ax.set_title("area to repair")
        ax = fig.add_subplot(133, projection="3d")
        ax.plot_surface(uu, vv, z[sl], cmap="viridis", linewidth=0)
        ax.set_title("3D surface")
        if _ask(fig, ("Correct", "Skip")) != "Correct":
            continue
        fixed = _fill_patch(holes[sl], method, 40000, du, dv)
        fig = plt.figure(figsize=(13, 6))
        fig.suptitle(f"defect {n} / {len(regions)}: original (left) and repaired (right)")
        for k, (data, title) in enumerate(((z[sl], "original"), (fixed, "repaired"))):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            ax.plot_surface(uu, vv, data, cmap="viridis", linewidth=0)
            ax.set_title(title)
        if _ask(fig, ("Validate", "Cancel")) == "Validate":
            z[sl] = fixed
            print(f"defect {n} repaired at iu = [{i0}, {i1}], iv = [{j0}, {j1}]")
    return road._derive(z=z)

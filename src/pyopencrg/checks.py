"""
Consistency checks of a road and readable summary.

:func:`check` runs the same kind of checks as the OpenCRG reference tools:

* grid size, increments (multiples of the minimal increment ``check_inc``);
* lateral positions sorted;
* range of heading, slope and banking values;
* NaN values (cross sections full of NaN, NaN inside a cross section);
* curvature: the center of a curve must not lie inside the road
  (otherwise two cross sections would intersect);
* options and modifiers values;
* consistency of the WGS84 positions.

Errors make the file unusable by other tools; warnings are worth a look.
"""

from __future__ import annotations

import os
import warnings as _warnings
from dataclasses import dataclass, field

import numpy as np

from .keywords import MODIFIER_KEYWORDS, MODS_BY_OFFSET, MODS_BY_REFPOINT, OPTION_DEFAULTS

__all__ = ["CheckReport", "CrgCheckError", "CrgCheckWarning", "check", "validate", "info_text"]


@dataclass
class CheckReport:
    """Result of :func:`check`.

    Attributes
    ----------
    errors : list of str
        Serious problems (other OpenCRG tools will refuse the road).
    warnings : list of str
        Possible problems.
    infos : list of str
        Information about what was checked.
    """

    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    infos: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no error and no warning was found."""
        return not self.errors and not self.warnings

    def __bool__(self):
        return self.ok

    def __str__(self):
        lines = []
        if self.ok:
            lines.append("OK: no problem found")
        for e in self.errors:
            lines.append(f"ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        for i in self.infos:
            lines.append(f"info:    {i}")
        return "\n".join(lines)

    def show(self):
        """Print the report."""
        print(self)


class CrgCheckError(ValueError):
    """Raised when a road read from a file does not pass the checks.

    The full report is in the ``report`` attribute.
    """

    def __init__(self, report, filename=""):
        self.report = report
        name = f"{filename}: " if filename else ""
        problems = report.errors or report.warnings
        super().__init__(
            f"{name}the file does not pass the OpenCRG checks:\n" + "\n".join(f"  - {p}" for p in problems)
            + "\nTo open it anyway (for instance to repair it), use crg.read(path, check=False), then"
              " e.g. road.trim_nan_borders(), road.fill_nan() or road.fill_holes().")


class CrgCheckWarning(UserWarning):
    """Warning emitted when a road read from a file has minor problems."""


def _multiple(value, inc, eps, tol):
    return abs(round(value / inc) * inc - value) <= max(eps * abs(value), tol)


def check(road) -> CheckReport:
    """Check a road and return a :class:`CheckReport`."""
    rep = CheckReport()
    E, W, I = rep.errors.append, rep.warnings.append, rep.infos.append
    for e in getattr(road, "read_errors", []):
        E(f"(reading) {e}")
    for w in road.read_warnings:
        W(f"(reading) {w}")

    # --- options
    opt = road.get_option
    for key, val in road.opts.items():
        if key not in OPTION_DEFAULTS:
            W(f"unknown option '{key}'")
    for key in ("border_mode_u", "border_mode_v"):
        val = opt(key)
        if val not in (0, 1, 2, 3, 4):
            E(f"option {key} = {val} must be 0, 1, 2, 3 or 4")
    if opt("refline_continuation") not in (0, 1):
        E("option refline_continuation must be 0 or 1")
    for key in ("border_smooth_ubeg", "border_smooth_uend"):
        if opt(key) < 0:
            E(f"option {key} must be positive")
    eps = opt("check_eps")
    if not 1e-6 <= eps <= 1e-2:
        E(f"option check_eps = {eps} must be in [1e-6, 1e-2]")
    cinc = opt("check_inc")
    tol = opt("check_tol")
    mininc = cinc * (1 - eps)

    # --- modifiers
    mods = road.effective_mods()
    for key in mods:
        if key not in MODIFIER_KEYWORDS:
            W(f"unknown modifier '{key}'")
    for key in ("scale_length", "scale_width"):
        if key in mods and mods[key] <= 0:
            E(f"modifier {key} must be positive")
    if "grid_nan_mode" in mods and mods["grid_nan_mode"] not in (0, 1, 2):
        E("modifier grid_nan_mode must be 0, 1 or 2")
    if mods.get("grid_nan_mode") == 0 and "grid_nan_offset" in mods:
        E("modifier grid_nan_offset is useless with grid_nan_mode = 0")
    if any(k in mods for k in MODS_BY_OFFSET) and any(k in mods for k in MODS_BY_REFPOINT):
        W("re-positioning modifiers 'by reference point' overwrite the 'by offset' ones")
    for a, b in (("refpoint_u", "refpoint_u_fraction"), ("refpoint_v", "refpoint_v_fraction")):
        if a in mods and b in mods:
            E(f"modifiers {a} and {b} cannot be both defined")
    if road.mods is None:
        I("no modifier block: OpenCRG default modifiers apply (NaN -> keep last value, start moved to origin)")

    # --- grid
    nu, nv = road.shape
    I(f"grid of {nu} x {nv} values, u in [{road.u_start:g}, {road.u_end:g}] m, v in [{road.v_min:g}, {road.v_max:g}] m")
    if road.u_inc < mininc:
        E(f"increment along u ({road.u_inc:g} m) is smaller than the minimal increment {cinc:g} m")
    for name, val in (("u increment", road.u_inc), ("u start", road.u_start), ("u end", road.u_end)):
        if not _multiple(val, cinc, eps, tol):
            W(f"{name} = {val:.9g} is not a multiple of {cinc:g} m")
    if road.v_inc is not None:
        if road.v_inc < mininc:
            E(f"lateral increment ({road.v_inc:g} m) is smaller than the minimal increment {cinc:g} m")
        elif not _multiple(road.v_inc, cinc, eps, tol):
            W(f"lateral increment {road.v_inc:.9g} is not a multiple of {cinc:g} m")
    bad_v = [vv for vv in road.v if not _multiple(vv, cinc, eps, tol)]
    if bad_v:
        W(f"{len(bad_v)} lateral position(s) are not multiples of {cinc:g} m (e.g. {bad_v[0]:.9g})")

    # --- reference line
    steps = np.hypot(np.diff(road.x), np.diff(road.y))
    err = np.max(np.abs(steps - road.u_inc))
    if err > max(tol, eps * road.u_inc):
        W(f"distance between reference line points differs from u_inc by up to {err:.3g} m")
    s = road.slope
    if np.any(np.abs(s) > 1):
        W(f"slope values above 100 % (max {np.max(np.abs(s)):.3g})")
    if np.any(np.abs(road.banking) > 1):
        W(f"banking values above 100 % (max {np.max(np.abs(road.banking)):.3g})")
    if abs(road.phi_offset) > np.pi * (1 + eps):
        W("reference_line_offset_phi is outside of [-pi, pi]")

    # --- curvature
    if not road.is_straight and road.curvature.size:
        rc = road.curvature
        bad = []
        for c in (rc.max(), rc.min()):
            if abs(c) > eps and road.v_min <= 1.0 / c <= road.v_max:
                bad.append(c)
        if bad:
            with np.errstate(divide="ignore"):  # zero curvature = infinite radius
                vec_r = 1.0 / np.concatenate(([rc[0]], rc, [rc[-1]]))
            inside = (vec_r >= road.v_min) & (vec_r <= road.v_max)
            zz = road.uv_to_z(road.u[inside], vec_r[inside])
            if np.all(np.isnan(zz)):
                I("curvature check: curve centers inside the road limits lie in NaN areas (accepted)")
            else:
                uf = road.u[inside][~np.isnan(zz)]
                W(f"curvature too high: curve center inside the road (radius {1 / max(bad, key=abs):.3g} m), "
                  f"cross sections intersect between u = {uf[0]:.3f} and u = {uf[-1]:.3f} m")

    # --- NaN
    full = np.zeros(nu, dtype=bool)
    inner = np.zeros(nu, dtype=bool)
    n_nan = 0
    block = max(1, 2_000_000 // nv)  # a few MB at a time, even for huge grids
    for r0 in range(0, nu, block):
        valid = ~np.isnan(road.z[r0:r0 + block])
        count = valid.sum(axis=1)
        n_nan += valid.size - int(count.sum())
        has = count > 0
        first = np.argmax(valid, axis=1)
        last = nv - 1 - np.argmax(valid[:, ::-1], axis=1)
        full[r0:r0 + block] = ~has
        inner[r0:r0 + block] = has & (count != last - first + 1)
    if n_nan:
        if full.any():
            E(f"{int(full.sum())} cross section(s) contain only NaN values (first at u = {road.u[np.argmax(full)]:.3f} m)")
        if inner.any():
            W(f"{int(inner.sum())} cross section(s) have NaN values between valid values "
              f"(first at u = {road.u[np.argmax(inner)]:.3f} m); other OpenCRG tools may refuse them, "
              f"use fill_holes() or fill_nan()")
        I(f"{100 * n_nan / (nu * nv):.3f} % of the grid values are NaN")

    # --- WGS84
    if road.lon_end is not None and road.lon_start is None:
        E("WGS84 end position defined without start position")
    if road.lon_start is not None:
        if not -180 <= road.lon_start <= 180 or not -90 <= road.lat_start <= 90:
            E("WGS84 start position out of range")
    if road.lon_start is not None and road.lon_end is not None and not road.mpro:
        from .wgs84 import distance
        dxy = np.hypot(road.x[-1] - road.x[0], road.y[-1] - road.y[0])
        dll = distance(road.lat_start, road.lon_start, road.lat_end, road.lon_end)[0]
        if abs(dxy - dll) > max(eps * (dxy + dll) / 2, opt("check_wgs")):
            W(f"WGS84 start/end distance ({dll:.3f} m) differs from the x/y distance ({dxy:.3f} m)")
    if road.alt_start is not None and road.alt_end is not None:
        dz = road.zr[-1] - road.zr[0]
        da = road.alt_end - road.alt_start
        if abs(dz - da) > max(eps * abs(dz + da) / 2, tol):
            W("WGS84 altitude difference is not consistent with the reference line elevation")
    if road.mpro:
        try:
            from .mapping import MapProjection
            MapProjection.from_keywords(road.mpro)
        except Exception as exc:  # noqa: BLE001
            E(f"map projection: {exc}")
        else:
            from .wgs84 import distance
            for name, i, lat, lon in (("start", 0, road.lat_start, road.lon_start),
                                      ("end", -1, road.lat_end, road.lon_end)):
                if lat is None or lon is None:
                    continue
                plat, plon, _ = road.xyz_to_llh(road.x[i], road.y[i], road.zr[i])
                d = distance(lat, lon, float(plat), float(plon))[0]
                if d > opt("check_pro"):
                    W(f"WGS84 {name} position is {d:.3f} m away from the projected x/y {name} position")
    return rep


def validate(road, strict: bool = False) -> CheckReport:
    """Check a road: raise :class:`CrgCheckError` on errors, warn once on warnings.

    With ``strict=True`` warnings are refused too.
    """
    rep = check(road)
    if rep.errors or (strict and rep.warnings):
        raise CrgCheckError(rep, road.filename)
    if rep.warnings:
        name = os.path.basename(road.filename) if road.filename else "road"
        shown = "\n".join(f"  - {w}" for w in rep.warnings[:5])
        more = f"\n  ... and {len(rep.warnings) - 5} more" if len(rep.warnings) > 5 else ""
        _warnings.warn(f"{name}: {len(rep.warnings)} warning(s), see road.check() for details:\n{shown}{more}",
                       CrgCheckWarning, stacklevel=3)
    return rep


def info_text(road) -> str:
    """Readable multi-line summary of a road."""
    lines = []
    name = road.filename or "(not saved)"
    lines.append(f"OpenCRG road: {name}")
    lines.append(f"  length          : {road.length:.3f} m  (u from {road.u_start:g} to {road.u_end:g} m, "
                 f"step {road.u_inc:g} m)")
    vinc = f"step {road.v_inc:g} m" if road.v_inc is not None else "uneven spacing"
    lines.append(f"  width           : {road.width:.3f} m  (v from {road.v_min:g} to {road.v_max:g} m, {vinc})")
    lines.append(f"  grid            : {road.nu} x {road.nv} = {road.nu * road.nv:,} values")
    kind = "straight" if road.is_straight else "curved"
    lines.append(f"  reference line  : {kind}, start (x, y, z) = ({road.x[0]:.3f}, {road.y[0]:.3f}, {road.zr[0]:.3f}), "
                 f"heading {np.degrees(road.phi_start):.3f} deg")
    lines.append(f"                    end   (x, y, z) = ({road.x[-1]:.3f}, {road.y[-1]:.3f}, {road.zr[-1]:.3f}), "
                 f"heading {np.degrees(road.phi_end):.3f} deg")
    if not road.is_straight:
        rc = road.curvature
        rmin = 1.0 / np.max(np.abs(rc)) if np.max(np.abs(rc)) > 0 else np.inf
        lines.append(f"  smallest radius : {rmin:.3f} m")
    s = road.slope
    lines.append(f"  slope           : {100 * s.min():.3f} % to {100 * s.max():.3f} %")
    b = road.banking
    lines.append(f"  banking         : {100 * b.min():.3f} % to {100 * b.max():.3f} %")
    z = road.z
    fin = np.isfinite(z)
    if fin.any():
        lines.append(f"  grid z          : {np.min(z[fin]):.4f} to {np.max(z[fin]):.4f} m, "
                     f"{100 * (1 - fin.mean()):.3f} % NaN")
    else:
        lines.append("  grid z          : only NaN values")
    if road.lon_start is not None:
        lines.append(f"  WGS84 start     : lat {road.lat_start:.8f}, lon {road.lon_start:.8f}")
    if road.lon_end is not None:
        lines.append(f"  WGS84 end       : lat {road.lat_end:.8f}, lon {road.lon_end:.8f}")
    if any(v != 0 for v in (road.x_offset, road.y_offset, road.z_offset, road.phi_offset)):
        lines.append(f"  offsets         : x {road.x_offset:g}, y {road.y_offset:g}, z {road.z_offset:g}, "
                     f"phi {road.phi_offset:g}")
    if road.mods is None:
        lines.append("  modifiers       : none in file (OpenCRG defaults apply)")
    elif road.mods:
        lines.append("  modifiers       : " + ", ".join(f"{k}={v:g}" for k, v in road.mods.items()))
    else:
        lines.append("  modifiers       : none")
    if road.opts:
        lines.append("  options         : " + ", ".join(f"{k}={v:g}" for k, v in road.opts.items()))
    if road.mpro:
        lines.append("  map projection  : " + ", ".join(f"{k}={v}" for k, v in road.mpro.items()))
    if road.comments:
        lines.append("  comments        :")
        lines += [f"      {c}" for c in road.comments[:20]]
        if len(road.comments) > 20:
            lines.append(f"      ... ({len(road.comments) - 20} more lines)")
    return "\n".join(lines)

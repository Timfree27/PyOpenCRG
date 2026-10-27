"""
Open an OpenCRG file once, then read parts of it: :func:`open` and :class:`CrgFile`.

``crg.read(path)`` reads a whole road (or a part of it) in one call. For
very large files, or when parts of the road are needed again and again
(a window sliding under the wheels of a simulated vehicle, a scan processed
piece by piece), the file can be opened once, like a Python file::

    import pyopencrg as crg

    with crg.open("scan.crg") as f:          # header and reference line, read once
        print(f.u_start, f.u_end)              # information without reading the grid
        part = f.read(u=(300, 330))            # a CrgRoad, read in a few milliseconds
        part2 = f.read(u=(330, 360))
        overview = f.read(step=8)              # light preview of the whole road

The parts keep the u values of the whole road, and their reference line is
exactly the one of the whole road: a part gives the same x, y, z as the whole
road. With ``f.read(u=..., apply_mods=True)`` the modifiers of the file are
computed once on the whole file, so the parts are placed exactly like in the
modified whole road.

For a simulation on a very large file, :meth:`CrgFile.evaluator` keeps only
a window of the road in memory and reads it again when the points move::

    with crg.open("track.crg") as f:
        ev = f.evaluator(window=40.0, margin=10.0, apply_mods=True)
        z, nx, ny, nz = ev.xy_to_z(x, y, with_normal=True)

Binary files (KRBI, KDBI) stay on the disk: only the requested part is
loaded. Text files (LRFI, LDFI) cannot be read by parts: their elevation
grid is loaded in memory when the file is opened. For very large text
files, convert them once to binary with ``crg.convert(path, new_path)``.
"""

from __future__ import annotations

import os

import numpy as np

from . import io as _io

__all__ = ["open", "CrgFile"]


class CrgFile:
    """An OpenCRG file opened for reading. Create it with :func:`pyopencrg.open`.

    Attributes
    ----------
    filename : str
        Name of the file.
    format : str
        Data format of the file (``KRBI``, ``KDBI``, ``LRFI`` or ``LDFI``).
    """

    def __init__(self, path, check=True):
        from .road import _reference_line

        self.filename = os.fspath(path)
        self._check = check
        content, ipl, layout = _io._decode_crg(self.filename)
        self.format = ipl.fmt
        nc = len(ipl.kd_def)
        full_cols = layout["full_cols"]
        if ipl.fmt in ("KRBI", "KDBI"):
            data = _io._binary_view(ipl.filename, ipl.data_offset, nc, ipl.fmt, ipl.errors, ipl.warnings)
            self._raise_if_corrupted(ipl, check)
            full = _io._take(data, slice(None), full_cols) if full_cols else np.zeros((data.shape[0], 0))
            self._cols = list(layout["z_cols"])
        elif ipl.fmt in ("LRFI", "LDFI"):
            data, full, _ = _io._ascii_stream(ipl.filename, ipl.data_offset, nc, 10 if ipl.fmt == "LRFI" else 20,
                                              slice(None), layout["z_cols"], full_cols, ipl.warnings, ipl.errors)
            self._raise_if_corrupted(ipl, check)
            self._cols = list(range(data.shape[1]))
        else:
            raise _io.CrgFormatError(f"{self.filename}: data format '{ipl.fmt}' is not supported")
        nrec = data.shape[0]
        if nrec < 2:
            raise _io.CrgFormatError(f"{self.filename}: at least 2 cross sections are required")
        content.warnings += ipl.warnings
        content.errors += ipl.errors
        u_inc = layout["u_inc"]
        if layout["inc_from_end"]:
            u_inc = (content.head["reference_line_end_u"] - layout["u_start"]) / (nrec - 1)
        content.u_start, content.u_inc = layout["u_start"], float(u_inc)
        content.n_records = nrec
        _io.set_reference_channels(content, full, layout)
        self._content = content
        self._v = layout["v"]
        self._data = data
        self._mods = None
        self._straight = None
        self._x, self._y, self._zr, self._b, self._warns = _reference_line(content, nrec)
        for a in (self._x, self._y, self._zr, self._b, self._v):
            a.setflags(write=False)

    @staticmethod
    def _raise_if_corrupted(ipl, check):
        if check and ipl.errors:
            from .checks import CheckReport, CrgCheckError
            raise CrgCheckError(CheckReport(errors=[f"(reading) {e}" for e in ipl.errors]), ipl.filename)

    # ------------------------------------------------------------------
    # file handling
    # ------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        """True once the file is closed."""
        return self._data is None

    def close(self) -> None:
        """Close the file. The roads already read stay usable."""
        self._data = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        state = "closed" if self.closed else "open"
        return (f"<CrgFile '{os.path.basename(self.filename)}' ({state}, {self.format}): {self.length:.3f} m, "
                f"grid {self.nu} x {self.nv}, u_inc={self.u_inc:g} m>")

    # ------------------------------------------------------------------
    # information (no grid data is read)
    # ------------------------------------------------------------------

    @property
    def nu(self) -> int:
        """Number of cross sections in the file."""
        return self._x.size

    @property
    def nv(self) -> int:
        """Number of long sections in the file."""
        return self._v.size

    @property
    def u_start(self) -> float:
        """u value of the first cross section (m)."""
        return self._content.u_start

    @property
    def u_inc(self) -> float:
        """Distance between two cross sections (m)."""
        return self._content.u_inc

    @property
    def u_end(self) -> float:
        """u value of the last cross section (m)."""
        return self.u_start + (self.nu - 1) * self.u_inc

    @property
    def length(self) -> float:
        """Length of the road (m)."""
        return (self.nu - 1) * self.u_inc

    @property
    def u(self) -> np.ndarray:
        """u value of every cross section (m)."""
        return self.u_start + np.arange(self.nu) * self.u_inc

    @property
    def v(self) -> np.ndarray:
        """Lateral position of every long section (m)."""
        return self._v

    @property
    def x(self) -> np.ndarray:
        """x position of the reference line at every cross section."""
        return self._x

    @property
    def y(self) -> np.ndarray:
        """y position of the reference line at every cross section."""
        return self._y

    @property
    def zr(self) -> np.ndarray:
        """Elevation of the reference line at every cross section."""
        return self._zr

    @property
    def banking(self) -> np.ndarray:
        """Banking at every cross section (m/m)."""
        return self._b

    @property
    def comments(self) -> list:
        """Comments of the file (``$CT`` block)."""
        return list(self._content.comments)

    @property
    def mods(self):
        """Modifiers of the file (``None`` if the file has no modifier block)."""
        return None if self._content.mods is None else dict(self._content.mods)

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def read(self, u=None, step=1, dtype="float64", check=None, apply_mods=False):
        """Read the whole road or a part of it.

        Parameters
        ----------
        u : (float, float), optional
            Read only the part between these two u values (m).
        step : int or (int, int), optional
            Keep one cross section (and one long section) out of ``step``.
        dtype : str, optional
            ``"float64"`` (default) or ``"float32"`` for the elevation grid.
        check : bool or "strict", optional
            Check of the part that is read; by default the one given to
            :func:`pyopencrg.open`.
        apply_mods : bool, optional
            Apply the modifiers of the file (like a simulator does). They are
            computed once on the **whole** file, so every part is placed
            exactly where it is in the modified whole road: two parts that
            overlap give the same x, y, z on their common area.

        Returns
        -------
        CrgRoad

        Notes
        -----
        A part is a road of its own: the border options of the file
        (``border_mode_u``, smoothing zones ``border_smooth_ubeg`` /
        ``border_smooth_uend``) apply at the ends of the part, and a point
        outside the part is extrapolated from its ends. To evaluate the
        whole road without loading it, use :meth:`evaluator`.

        Examples
        --------
        >>> with crg.open("scan.crg") as f:                           # doctest: +SKIP
        ...     part = f.read(u=(300, 330), apply_mods=True)          # same place as in the whole road
        """
        su, sv = _io._steps(step)
        i0, i1 = 0, self.nu
        if u is not None:
            if u[0] is not None:
                i0 = max(0, int(np.ceil((u[0] - self.u_start) / self.u_inc - 1e-9)))
            if u[1] is not None:
                i1 = min(self.nu, int(np.floor((u[1] - self.u_start) / self.u_inc + 1e-9)) + 1)
        if len(range(i0, i1, su)) < 2:
            raise ValueError(f"less than 2 cross sections between u = {u[0]} and {u[1]} "
                             f"(the road goes from {self.u_start:g} to {self.u_end:g} m)")
        return self._read(i0, i1, su, sv, dtype, check, apply_mods)

    def evaluator(self, window=40.0, margin=10.0, apply_mods=False, dtype="float64", check=False):
        """Evaluator that reads the road by windows, for simulations on very large files.

        Only a window of the road (``window`` metres) is in memory. When a
        point comes closer than ``margin`` to an end of the window, the
        window is read again around the points (a few milliseconds). The
        results are the ones of the whole road: positions are searched on
        the reference line of the whole file, and the borders, smoothing
        zones and closed tracks are the ones of the whole road. A point is
        never evaluated outside the loaded window.

        Keep the file open while the evaluator is used.

        Parameters
        ----------
        window : float, optional
            Length of road read at a time (m). It grows if the points are
            spread over a longer distance.
        margin : float, optional
            The window is read again when a point is closer than this to one
            of its ends (m). Must be less than half of ``window``.
        apply_mods : bool, optional
            Evaluate the road with the modifiers of the file applied (like a
            simulator does); see :meth:`read`.
        dtype : str, optional
            ``"float64"`` (default) or ``"float32"`` for the elevation grid of
            the window.
        check : bool or "strict", optional
            Check every window when it is read (off by default: the file was
            already checked when it was opened).

        Returns
        -------
        pyopencrg.evaluator.FileEvaluator

        Examples
        --------
        >>> with crg.open("track.crg") as f:                                    # doctest: +SKIP
        ...     ev = f.evaluator(window=40.0, margin=10.0, apply_mods=True)
        ...     for step in range(n_steps):
        ...         z, nx, ny, nz = ev.xy_to_z(x, y, with_normal=True)
        """
        from .evaluator import FileEvaluator
        return FileEvaluator(self, window, margin, apply_mods, dtype, check)

    def _geometry(self, apply_mods):
        """Road with the reference line of the whole file (its grid is not used)."""
        from .road import _assemble
        if apply_mods:
            self._modifiers()
            return self._mods[0]
        return _assemble(self._content, np.zeros((self.nu, 2)), np.array(self._v[[0, -1]]), None,
                         self._x, self._y, self._zr, self._b, self._warns)

    def _modifiers(self):
        if self._mods is None:  # computed once, on the whole file
            from .mods import _file_mods
            self._mods = _file_mods(self._geometry(False), np.array(self._v),
                                    lambda i0, i1: _io._take(self._data, slice(i0, i1), self._cols))
        return self._mods

    def _read(self, i0, i1, su, sv, dtype, check, apply_mods):
        """Part made of the rows ``i0:i1:su`` and the columns ``::sv`` of the file."""
        from .keywords import OPTION_DEFAULTS
        from .road import _assemble, _grid_dtype, _straight_line

        if self.closed:
            raise ValueError("read of a closed CrgFile")
        rows = np.arange(self.nu)[i0:i1:su]
        keep_v = np.arange(0, self.nv, sv)
        cols = [self._cols[j] for j in keep_v]
        z = _io._take(self._data, slice(i0, i1, su), cols, dtype=_grid_dtype(dtype))
        if self._straight is None:
            self._straight = _straight_line(self._x, self._y, self._content.opts.get("check_eps",
                                                                                     OPTION_DEFAULTS["check_eps"]))
        road = _assemble(self._content, z, np.array(self._v[keep_v]), rows, self._x, self._y, self._zr, self._b,
                         self._warns, self._straight)
        check = self._check if check is None else check
        if check:
            from .checks import validate
            validate(road, strict=(check == "strict"))
        if apply_mods:
            road = self._modified(road, rows, su)
        return road

    def _modified(self, part, rows, su):
        """The part ``part`` (rows ``rows`` of the file) with the modifiers of the whole file."""
        from .edit import _fill_nan_grid
        from .road import CrgRoad, _set_part_line

        ref, zs, ws, nan = self._modifiers()
        z = part.z if zs == 1.0 else part.z * part.z.dtype.type(zs)  # a new array, not used elsewhere
        if nan is not None:
            z = _fill_nan_grid(z, part.v * ws, *nan)
        new = CrgRoad._from_state(z, part.v * ws, ref.u_start + rows[0] * ref.u_inc, ref.u_inc * su, ref.x[rows],
                                  ref.y[rows], ref.zr[rows], ref.banking[rows], template=ref)
        new.read_warnings = list(part.read_warnings)
        new.read_errors = list(part.read_errors)
        new._part = part._part
        if new._part:
            _set_part_line(new, ref.x, ref.y, rows, ref.is_straight)
        return new


def open(path, check=True) -> CrgFile:  # noqa: A001 - same name as the built-in open, on purpose
    """Open an OpenCRG file to read it once or by parts.

    The header and the reference line are read right away; the elevation
    grid is read by :meth:`CrgFile.read`. Use it in a ``with`` block, like a
    Python file.

    Parameters
    ----------
    path : str or path-like
        The ``.crg`` file.
    check : bool or "strict", optional
        ``True`` (default): refuse corrupted files and check every part that
        is read. ``False``: no check.

    Returns
    -------
    CrgFile

    Examples
    --------
    >>> with crg.open("scan.crg") as f:            # doctest: +SKIP
    ...     for u0 in range(0, int(f.length), 100):
    ...         part = f.read(u=(u0, u0 + 100))   # 100 m at a time
    ...         print(u0, part.stats()["max"])
    """
    return CrgFile(path, check)

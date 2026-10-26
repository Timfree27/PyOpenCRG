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
exactly the one of the whole road.

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

    def read(self, u=None, step=1, dtype="float64", check=None):
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

        Returns
        -------
        CrgRoad
        """
        from .road import _assemble, _grid_dtype

        if self.closed:
            raise ValueError("read of a closed CrgFile")
        su, sv = _io._steps(step)
        i0, i1 = 0, self.nu
        if u is not None:
            if u[0] is not None:
                i0 = max(0, int(np.ceil((u[0] - self.u_start) / self.u_inc - 1e-9)))
            if u[1] is not None:
                i1 = min(self.nu, int(np.floor((u[1] - self.u_start) / self.u_inc + 1e-9)) + 1)
        rows = np.arange(self.nu)[i0:i1:su]
        if rows.size < 2:
            raise ValueError(f"less than 2 cross sections between u = {u[0]} and {u[1]} "
                             f"(the road goes from {self.u_start:g} to {self.u_end:g} m)")
        keep_v = np.arange(0, self.nv, sv)
        cols = [self._cols[j] for j in keep_v]
        z = _io._take(self._data, slice(i0, i1, su), cols, dtype=_grid_dtype(dtype))
        road = _assemble(self._content, z, np.array(self._v[keep_v]), rows, self._x, self._y, self._zr, self._b,
                         self._warns)
        check = self._check if check is None else check
        if check:
            from .checks import validate
            validate(road, strict=(check == "strict"))
        return road


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

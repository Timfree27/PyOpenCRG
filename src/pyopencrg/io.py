"""
Reading and writing of OpenCRG files.

OpenCRG files are "IPLOS" files: a text header made of named blocks
(``$CT``, ``$ROAD_CRG``, ``$KD_DEFINITION``...), a separator line made of
``$`` characters, then the data, one record per cross section of the road.

The data can be stored in four formats:

========  ==================================================
``KRBI``  binary, float32 (smallest and fastest, default)
``KDBI``  binary, float64
``LRFI``  text, single precision (10 characters per value)
``LDFI``  text, double precision (20 characters per value)
========  ==================================================

Everything here is vectorised with numpy so that files of several hundred
megabytes can be read and written in a few seconds.

Most users only need :func:`pyopencrg.read` and :meth:`pyopencrg.CrgRoad.save`.
The low level functions :func:`read_ipl` and :func:`write_ipl` are available
for people who want to work with other IPLOS files.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .keywords import (
    FORMATS,
    MODIFIER_KEYWORDS,
    MPRO_KEYWORDS,
    OPTION_DEFAULTS,
    OTHER_TOOL_OPTIONS,
    ROAD_CRG_KEYWORDS,
    SEPARATOR,
)

__all__ = ["IplData", "read_ipl", "write_ipl", "CrgFileContent", "read_crg_file", "format_values"]

_MAX_HEADER_LINE = 72


class CrgFormatError(ValueError):
    """Raised when a file is not a valid OpenCRG / IPLOS file."""


class CrgCorruptedError(CrgFormatError):
    """Raised when the data of a file is corrupted; the problems are listed in ``errors``."""

    def __init__(self, filename, errors):
        self.errors = list(errors)
        super().__init__(f"{filename}: corrupted data:\n" + "\n".join(f"  - {e}" for e in self.errors)
                         + "\nUse crg.read(path, check=False) to open it anyway.")


# ===========================================================================
# Low level IPLOS reading
# ===========================================================================

@dataclass
class IplData:
    """Raw content of an IPLOS file.

    Attributes
    ----------
    struct : list of str
        Header lines, except the ``$KD_DEFINITION`` block.
    kd_ind : list of str
        Virtual channel definitions (``U:`` lines), e.g. ``"reference line u,m,0,0.01"``.
    kd_def : list of str
        Data channel definitions (``D:`` lines), e.g. ``"long section 1,m"``.
    kd_oth : list of str
        Other lines of the ``$KD_DEFINITION`` block.
    fmt : str
        Data format (``KRBI``, ``KDBI``, ``LRFI`` or ``LDFI``).
    data : numpy.ndarray or None
        Data array of shape (number of records, number of channels), float64.
    filename : str
        Name of the file that was read.
    warnings : list of str
        Minor problems found while reading.
    errors : list of str
        Serious problems: truncated or unreadable data (the data is not complete).
    """

    struct: list = field(default_factory=list)
    kd_ind: list = field(default_factory=list)
    kd_def: list = field(default_factory=list)
    kd_oth: list = field(default_factory=list)
    fmt: str = "KRBI"
    data: np.ndarray | None = None
    filename: str = ""
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    data_offset: int = 0


def _first_token(line: str) -> str:
    """First word of a header line, cut at a blank or a '!' comment."""
    return re.split(r"[ !\t]", line, maxsplit=1)[0]


def _is_separator(line: str) -> bool:
    s = line.strip()
    return len(s) >= 4 and set(s) == {"$"}


def _scan_header(path: str):
    """Read header lines until the ``$$$$`` separator.

    Returns (lines, byte offset of the data, separator found).
    """
    lines = []
    offset = 0
    state = 0  # 0 = outside of a block, 1 = inside a block
    with open(path, "rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                return lines, offset, False
            offset += len(raw)
            line = raw.decode("latin-1").rstrip("\r\n")
            if state == 0:
                if line.startswith("$$"):
                    return lines, offset, True
                if line.startswith("$") and _first_token(line) != "$":
                    state = 1
            else:
                if _is_separator(line):
                    # tolerated: missing block end marker before the separator
                    return lines, offset, True
                if line.startswith("$") and _first_token(line) == "$":
                    state = 0
            lines.append(line)


def cut_block(lines: list, name: str):
    """Extract the named block from a list of header lines.

    Parameters
    ----------
    lines : list of str
        Header lines.
    name : str
        Block name without the leading ``$`` (case insensitive), e.g. ``"ROAD_CRG"``.

    Returns
    -------
    block : list of str or None
        Lines inside the block (``None`` if the block does not exist).
    rest : list of str
        All other lines.
    """
    target = "$" + name.upper()
    block = None
    rest = []
    state = 0
    for line in lines:
        if state == 0:
            if line.startswith("$") and _first_token(line).upper() == target:
                state = 1
                block = []
            else:
                rest.append(line)
        elif state == 1:
            if line.startswith("$$"):
                block.append(line[1:])  # escaped '$'
            elif line.startswith("$"):
                state = 2
                if _first_token(line) != "$":
                    rest.append(line)  # tolerated: next block starts without end marker
            else:
                block.append(line)
        else:
            rest.append(line)
    return block, rest


def _clean_line(line: str) -> str | None:
    """Remove comments of a header line. Returns None for comment lines."""
    if line.startswith("*"):
        return None
    return line.split("!", 1)[0].strip()


def _parse_kd_definition(block: list, ipl: IplData) -> None:
    for line in block:
        hc = _clean_line(line)
        if not hc:
            continue
        tag = hc[:2].upper()
        if tag == "#:":
            ipl.fmt = hc[2:].strip().upper()
        elif tag == "U:":
            ipl.kd_ind.append(hc[2:].strip())
        elif tag == "D:":
            ipl.kd_def.append(hc[2:].strip())
        else:
            ipl.kd_oth.append(hc)


def read_ipl(path: str | os.PathLike, meta_only: bool = False) -> IplData:
    """Read an IPLOS file (the container format used by OpenCRG).

    Parameters
    ----------
    path : str or path-like
        File to read.
    meta_only : bool, optional
        If True, only the header is read (very fast, even for huge files).

    Returns
    -------
    IplData
        The raw content of the file.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {path}")
    lines, offset, found = _scan_header(path)
    ipl = IplData(filename=path, data_offset=offset)
    kd_block, ipl.struct = cut_block(lines, "KD_DEFINITION")
    if not found or kd_block is None:
        if meta_only and kd_block is None:
            return ipl
        if kd_block is None:
            raise CrgFormatError(f"{path}: no $KD_DEFINITION block found (is this an OpenCRG file?)")
    _parse_kd_definition(kd_block, ipl)
    nc = len(ipl.kd_def)
    if nc == 0:
        raise CrgFormatError(f"{path}: no data channel (D:) defined in $KD_DEFINITION")
    if meta_only:
        return ipl
    if not found:
        raise CrgFormatError(f"{path}: no separator line ($$$$...) between header and data")

    ipl.data = read_ipl_data(ipl)[0]
    return ipl


def _row_mask(idx, rows):
    """Boolean mask of the record indices ``idx`` selected by the slice ``rows``."""
    start = rows.start or 0
    step = rows.step or 1
    mask = (idx >= start) & ((idx - start) % step == 0)
    if rows.stop is not None:
        mask &= idx < rows.stop
    return mask


_UTF8_REPLACEMENT = b"\xef\xbf\xbd"


def _binary_view(path, offset, nc, fmt, errors):
    """Memory mapped (nrec, nc) view of the binary data; nothing is loaded in memory."""
    dtype = np.dtype(">f4" if fmt == "KRBI" else ">f8")
    per_block = 80 // dtype.itemsize
    nbytes = os.path.getsize(path) - offset
    with open(path, "rb") as fh:
        fh.seek(offset)
        sample = fh.read(min(nbytes, 4 << 20))
    if sample and sample.count(_UTF8_REPLACEMENT) * 3 > 0.01 * len(sample):
        errors.append("the binary data has been converted to text (it is full of the UTF-8 replacement "
                      "character U+FFFD): the file was opened and saved again by a text editor or by a "
                      "program reading it as text. The original values are lost, use an original copy")
    if nbytes % dtype.itemsize:
        errors.append(f"size of the binary data is not a multiple of {dtype.itemsize} bytes (corrupted file?)")
    count = nbytes // dtype.itemsize
    if count % per_block:
        errors.append("unexpected end of binary data (size is not a multiple of 80 bytes: truncated file?)")
    nr = count // nc
    if nr == 0:
        return np.zeros((0, nc))
    view = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(nr, nc))
    rest = count - nr * nc
    if rest:
        tail = np.fromfile(path, dtype=dtype, count=rest, offset=offset + nr * nc * dtype.itemsize)
        if not np.all(np.isnan(tail)):
            errors.append("binary data is not correctly padded with NaN (truncated or corrupted file?)")
    while nr > 0 and np.all(np.isnan(view[nr - 1])):  # NaN records appended as padding
        nr -= 1
    return view[:nr]


def _take(view, rows, cols, block=8192):
    """Copy ``view[rows][:, cols]`` to a float64 array, a few thousand records at a time."""
    idx = np.arange(view.shape[0])[rows]
    out = np.empty((idx.size, len(cols)), dtype=np.float64)
    for k in range(0, idx.size, block):
        sel = idx[k:k + block]
        out[k:k + sel.size] = view[sel[0]:sel[-1] + 1][sel - sel[0]][:, cols]
    return out


def _ascii_stream(path, offset, nc, nchar, rows, cols, full_cols, warns, errors, chunk_bytes=64 << 20):
    """Parse a text data part chunk by chunk, keeping only the selected records."""
    vpl = 80 // nchar
    lpr = -(-nc // vpl)
    size = os.path.getsize(path)
    pos = offset
    leftover = np.zeros(0, dtype=np.uint8)
    rec0 = 0
    sel_parts, full_parts = [], []
    while pos < size or leftover.size:
        n = min(chunk_bytes, size - pos)
        buf = np.fromfile(path, dtype=np.uint8, count=n, offset=pos) if n > 0 else np.zeros(0, np.uint8)
        pos += n
        if leftover.size:
            buf = np.concatenate((leftover, buf))
        if pos < size:
            nl = np.flatnonzero(buf == 10)
            nrec = nl.size // lpr
            if nrec == 0:
                leftover = buf
                continue
            cut = nl[nrec * lpr - 1] + 1
            part, leftover = buf[:cut], buf[cut:]
        else:
            part, leftover = buf, np.zeros(0, dtype=np.uint8)
        data = parse_ascii_data(part, nc, nchar, warns, errors)
        idx = np.arange(rec0, rec0 + data.shape[0])
        rec0 += data.shape[0]
        sel_parts.append(data[_row_mask(idx, rows)][:, cols])
        full_parts.append(data[:, list(full_cols)])
    if not sel_parts:
        return np.zeros((0, len(cols))), np.zeros((0, len(full_cols))), 0
    return np.concatenate(sel_parts), np.concatenate(full_parts), rec0


def read_ipl_data(ipl, rows=None, columns=None, full_columns=(), stop_on_error=False):
    """Read the data of an IPLOS file whose header was read with ``read_ipl(path, meta_only=True)``.

    Only the requested part is loaded in memory, so that very large files
    can be handled.

    Parameters
    ----------
    ipl : IplData
    rows : slice, optional
        Records to keep (default: all).
    columns : list of int, optional
        Channels to keep for the selected records (default: all).
    full_columns : list of int
        Channels to read for **all** records (e.g. the reference line).
    stop_on_error : bool
        Raise :class:`CrgFormatError` as soon as the binary data is found to be
        corrupted, before anything is loaded in memory.

    Returns
    -------
    selected : numpy.ndarray (selected records, len(columns))
    full : numpy.ndarray (all records, len(full_columns))
    n_records : int
        Total number of records in the file.
    """
    rows = slice(None) if rows is None else rows
    nc = len(ipl.kd_def)
    columns = list(range(nc)) if columns is None else list(columns)
    full_columns = list(full_columns)
    fmt = ipl.fmt
    if fmt in ("KRBI", "KDBI"):
        view = _binary_view(ipl.filename, ipl.data_offset, nc, fmt, ipl.errors)
        if stop_on_error and ipl.errors:
            del view
            raise CrgCorruptedError(ipl.filename, ipl.errors)
        sel = _take(view, rows, columns)
        full = _take(view, slice(None), full_columns) if full_columns else np.zeros((view.shape[0], 0))
        n = view.shape[0]
        del view
        return sel, full, n
    if fmt in ("LRFI", "LDFI"):
        return _ascii_stream(ipl.filename, ipl.data_offset, nc, 10 if fmt == "LRFI" else 20, rows, columns,
                             full_columns, ipl.warnings, ipl.errors)
    raise CrgFormatError(f"{ipl.filename}: data format '{fmt}' is not supported (use one of {list(FORMATS)})")


def _fields_to_float(fields: np.ndarray, nchar: int, warns: list, errors: list) -> np.ndarray:
    """Convert fixed width text fields (uint8 array [..., nchar]) to floats."""
    f = np.ascontiguousarray(fields)
    d_exp = (f == 68) | (f == 100)  # Fortran style exponent 'D'
    if d_exp.any():
        f[d_exp] = 69
    stars = np.any(f == 42, axis=-1)
    blank = np.all(f == 32, axis=-1)
    if blank.any():
        warns.append(f"{int(blank.sum())} empty (white space) value(s) read as NaN")
    s = f.view(f"S{nchar}")[..., 0]
    bad = stars | blank
    if bad.any():
        s[bad] = b"nan"
    try:
        return s.astype(np.float64)
    except ValueError:
        out = np.empty(s.shape, dtype=np.float64)
        nbad = 0
        for i, item in enumerate(s.ravel()):
            try:
                out.flat[i] = float(item)
            except ValueError:
                out.flat[i] = np.nan
                nbad += 1
        errors.append(f"{nbad} unreadable value(s) in the data (read as NaN)")
        return out


def parse_ascii_data(buf: np.ndarray, nc: int, nchar: int, warns: list | None = None,
                     errors: list | None = None) -> np.ndarray:
    """Parse the fixed width text data of a LRFI / LDFI file.

    Each record (cross section) starts on a new line; lines hold
    ``80 // nchar`` values of ``nchar`` characters.

    Parameters
    ----------
    buf : numpy.ndarray of uint8
        Raw bytes of the data part of the file.
    nc : int
        Number of channels (values per record).
    nchar : int
        Width of one value (10 for LRFI, 20 for LDFI).
    warns, errors : list, optional
        Minor / serious problems found are appended to these lists.

    Returns
    -------
    numpy.ndarray
        Array of shape (records, nc).
    """
    if warns is None:
        warns = []
    if errors is None:
        errors = []
    vpl = 80 // nchar
    lpr = -(-nc // vpl)  # lines per record
    nl = np.flatnonzero(buf == 10)
    starts = np.concatenate(([0], nl + 1))
    ends = np.concatenate((nl, [buf.size]))
    lengths = ends - starts
    has_cr = np.zeros(lengths.shape, dtype=bool)
    pos = lengths > 0
    has_cr[pos] = buf[ends[pos] - 1] == 13
    lengths = lengths - has_cr
    nonempty = np.flatnonzero(lengths > 0)
    last = nonempty[-1] + 1 if nonempty.size else 0
    starts, lengths = starts[:last], lengths[:last]
    nlines = starts.size
    nrec = nlines // lpr
    if nlines % lpr:
        errors.append("unexpected end of text data (last record is incomplete: truncated file?)")
    out = np.empty((nrec, nc), dtype=np.float64)
    cols = np.arange(80)
    chunk = max(1, 100_000 // lpr)
    for r0 in range(0, nrec, chunk):
        r1 = min(nrec, r0 + chunk)
        st = starts[r0 * lpr: r1 * lpr]
        ln = np.minimum(lengths[r0 * lpr: r1 * lpr], 80)
        mask = cols[None, :] < ln[:, None]
        mat = np.full((st.size, 80), 32, dtype=np.uint8)
        mat[mask] = buf[(st[:, None] + cols[None, :])[mask]]
        fields = mat.reshape(r1 - r0, lpr * vpl, nchar)[:, :nc, :]
        out[r0:r1] = _fields_to_float(fields, nchar, warns, errors)
    return out


# ===========================================================================
# Low level IPLOS writing
# ===========================================================================

def format_values(x, n: int) -> np.ndarray:
    """Format numbers as text fields of exactly ``n`` characters.

    Fixed point notation is used whenever possible, with as many digits as
    the width allows (``0.5`` -> ``".500000000"``); very large numbers use
    the exponent notation. NaN and infinite values become ``*`` characters.

    Parameters
    ----------
    x : array_like
        Numbers to format.
    n : int
        Width of each field.

    Returns
    -------
    numpy.ndarray
        uint8 array of shape (len(x), n) holding the characters.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    num = x.size
    out = np.full((num, n), 42, dtype=np.uint8)
    finite = np.isfinite(x)
    neg = x < 0
    a = np.abs(np.where(finite, x, 0.0))
    minus = neg.astype(np.int64)
    ndig = n - 1 - minus
    with np.errstate(divide="ignore"):
        e = np.floor(np.log10(np.where(a > 0, a, 1.0))).astype(np.int64) + 1
    intd = np.where(a > 0, np.maximum(e, 0), 0)
    frac = ndig - intd
    small = finite & (a > 0) & (e < -2)  # |x| < 0.001: exponent notation keeps the digits
    ok = finite & (frac >= 0) & ~small
    mant = np.zeros(num, dtype=np.float64)
    idx = np.flatnonzero(ok)
    if idx.size:
        mm = np.rint(a[idx] * 10.0 ** frac[idx])
        carry = mm >= 10.0 ** ndig[idx]
        if carry.any():  # rounding produced one more digit, e.g. 9.9999 -> 10.000
            ic = idx[carry]
            intd[ic] += 1
            frac[ic] -= 1
            good = frac[ic] >= 0
            mm[carry] = np.where(good, np.rint(a[ic] * 10.0 ** np.maximum(frac[ic], 0)), 0.0)
            ok[ic[~good]] = False
        mant[idx] = mm
    rows = np.flatnonzero(ok)
    if rows.size:
        m = mant[rows].astype(np.int64)
        pow10 = 10 ** np.arange(n - 2, -1, -1, dtype=np.int64)
        digits = (m[:, None] // pow10[None, :]) % 10  # (rows, n-1), right aligned
        c = np.arange(n)[None, :]
        p = (minus[rows] + intd[rows])[:, None]  # column of the decimal point
        di = np.clip(c - (c > p), 0, n - 2)
        chars = (np.take_along_axis(digits, di, axis=1) + 48).astype(np.uint8)
        chars[c.repeat(rows.size, 0) == p] = 46  # '.'
        is_minus = (c == 0) & (minus[rows][:, None] == 1)
        chars[is_minus] = 45  # '-'
        out[rows] = chars
    rows = np.flatnonzero(small)
    if rows.size:  # integer mantissa followed by e-XX, e.g. '66667e-09'
        nd = n - minus[rows] - 4
        ex = e[rows] - nd
        m = np.rint(a[rows] / 10.0 ** ex)
        carry = m >= 10.0 ** nd
        ex[carry] += 1
        m[carry] = np.rint(a[rows][carry] / 10.0 ** ex[carry])
        tiny = ex <= -100
        m[tiny] = 0
        ex[tiny] = -99
        pow10 = 10 ** np.arange(n - 5, -1, -1, dtype=np.int64)
        digits = (m.astype(np.int64)[:, None] // pow10[None, :]) % 10  # n-4 digits, right aligned
        chars = np.empty((rows.size, n), dtype=np.uint8)
        neg_r = minus[rows] == 1
        chars[:, : n - 4] = digits + 48
        chars[neg_r, 0] = 45  # the leading digit is 0 when there is a minus sign
        chars[:, n - 4] = 101  # 'e'
        chars[:, n - 3] = 45
        chars[:, n - 2] = 48 + (-ex) // 10
        chars[:, n - 1] = 48 + (-ex) % 10
        out[rows] = chars
    for i in np.flatnonzero(finite & ~ok & ~small):  # very large values: exponent notation
        for prec in range(n, -1, -1):
            s = f"{x[i]:.{prec}E}"
            if len(s) <= n:
                break
        out[i] = np.frombuffer(s.rjust(n)[:n].encode("ascii"), dtype=np.uint8)
    return out


def _ascii_gather_map(nc: int, nchar: int) -> np.ndarray:
    """Byte layout of one record: values, with a newline every 80 // nchar values."""
    vpl = 80 // nchar
    newline = nc * nchar
    gmap = []
    for line_start in range(0, nc, vpl):
        for f in range(line_start, min(nc, line_start + vpl)):
            gmap.extend(range(f * nchar, (f + 1) * nchar))
        gmap.append(newline)
    return np.asarray(gmap, dtype=np.int64)


def _write_ascii(fh, data: np.ndarray, nchar: int) -> None:
    nr, nc = data.shape
    gmap = _ascii_gather_map(nc, nchar)
    chunk = max(1, 400_000 // max(nc, 1))
    for r0 in range(0, nr, chunk):
        block = data[r0: r0 + chunk]
        flat = block.ravel()
        fields = np.empty((flat.size, nchar), dtype=np.uint8)
        fields[:, 0] = 32
        fields[:, 1:] = format_values(flat, nchar - 1)
        fields[~np.isfinite(flat)] = 42
        rec = np.empty((block.shape[0], nc * nchar + 1), dtype=np.uint8)
        rec[:, :-1] = fields.reshape(block.shape[0], nc * nchar)
        rec[:, -1] = 10
        fh.write(np.ascontiguousarray(rec[:, gmap]).tobytes())


def write_ipl(path, struct: list, kd_ind: list, kd_def: list, data: np.ndarray,
              fmt: str = "KRBI", kd_oth: list | None = None) -> None:
    """Write an IPLOS file.

    Parameters
    ----------
    path : str or path-like
        File to write (overwritten if it exists).
    struct : list of str
        Header lines (blocks ``$NAME ... $``).
    kd_ind, kd_def : list of str
        Virtual (``U:``) and data (``D:``) channel definitions.
    data : numpy.ndarray
        Array of shape (records, channels).
    fmt : str
        ``KRBI`` (default), ``KDBI``, ``LRFI`` or ``LDFI``.
    kd_oth : list of str, optional
        Other lines for the ``$KD_DEFINITION`` block.
    """
    fmt = fmt.upper()
    if fmt not in FORMATS:
        raise ValueError(f"unknown format '{fmt}', use one of {list(FORMATS)}")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(kd_def):
        raise ValueError("data must be a 2D array with one column per channel definition")
    head = list(struct)
    head.append("$KD_DEFINITION")
    head.append(f"#:{fmt}")
    head += [f"U:{s}" for s in kd_ind]
    head += [f"D:{s}" for s in kd_def]
    head += list(kd_oth or [])
    head.append("$")
    head.append(SEPARATOR)
    text = "\n".join(head) + "\n"
    with open(os.fspath(path), "wb") as fh:
        fh.write(text.encode("latin-1", errors="replace"))
        if fmt in ("KRBI", "KDBI"):
            dtype = ">f4" if fmt == "KRBI" else ">f8"
            per_block = 20 if fmt == "KRBI" else 10
            fh.write(np.ascontiguousarray(data, dtype=dtype).tobytes())
            pad = (-data.size) % per_block
            if pad:
                fh.write(np.full(pad, np.nan, dtype=dtype).tobytes())
        else:
            _write_ascii(fh, data, 10 if fmt == "LRFI" else 20)


# ===========================================================================
# OpenCRG level
# ===========================================================================

@dataclass
class CrgFileContent:
    """Decoded content of an OpenCRG file, before it is turned into a road.

    This is an intermediate object used by :func:`pyopencrg.read`.
    """

    filename: str = ""
    comments: list = field(default_factory=list)
    head: dict = field(default_factory=dict)
    opts: dict = field(default_factory=dict)
    other_tool_opts: dict = field(default_factory=dict)
    mods: dict | None = None
    mpro: dict | None = None
    extra: list = field(default_factory=list)
    u_start: float = 0.0
    u_inc: float = 0.01
    v: np.ndarray | None = None
    z: np.ndarray | None = None
    phi: np.ndarray | None = None
    slope: np.ndarray | None = None
    banking: np.ndarray | None = None
    rows: np.ndarray | None = None
    n_records: int = 0
    fmt: str = "KRBI"
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _parse_keyword_block(block: list, known, numeric_only=True, warns=None):
    """Parse ``keyword = value`` lines. Unknown keywords are reported."""
    values = {}
    for line in block or []:
        hc = _clean_line(line)
        if not hc or "=" not in hc:
            continue
        key, val = hc.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if key not in known:
            if warns is not None:
                warns.append(f"unknown keyword '{key}' ignored")
            continue
        if not numeric_only and key.endswith("_nm"):
            m = re.search(r"'([^']*)'", val)
            values[key] = (m.group(1) if m else val).strip()
            continue
        try:
            values[key] = float(val.split()[0].replace("D", "E").replace("d", "e"))
        except (ValueError, IndexError):
            if warns is not None:
                warns.append(f"cannot read value of '{key}': '{val}'")
    return values


def _resolve_file_reference(block, warns):
    parts = []
    for line in block:
        if line.startswith("*"):
            continue
        parts.append(line.split("!", 1)[0].strip())
    name = "".join(parts)
    return os.path.expandvars(name)


_KNOWN_BLOCKS = ("CT", "ROAD_CRG", "ROAD_CRG_OPTS", "ROAD_CRG_MODS", "ROAD_CRG_MPRO", "ROAD_CRG_FILE")


def _steps(step):
    su, sv = (step, step) if np.isscalar(step) else (int(step[0]), int(step[1]))
    if su < 1 or sv < 1 or su != int(su) or sv != int(sv):
        raise ValueError("step must be a positive integer or a pair of positive integers")
    return int(su), int(sv)


def read_crg_file(path: str | os.PathLike, u=None, step=1, stop_on_error=False) -> CrgFileContent:
    """Read an OpenCRG file and decode its header and channels.

    Use :func:`pyopencrg.read` instead, which returns a ready to use
    :class:`pyopencrg.CrgRoad`.

    Parameters
    ----------
    u : (float, float), optional
        Read only the cross sections between these u values (m).
    step : int or (int, int)
        Keep one cross section out of ``step`` (and one long section out of
        ``step``), to get a light preview of a very large file.
    stop_on_error : bool
        Refuse corrupted binary data before loading it.
    """
    path = os.fspath(path)
    su, sv = _steps(step)
    ipl = read_ipl(path, meta_only=True)
    out = CrgFileContent(filename=path, fmt=ipl.fmt)
    struct = ipl.struct

    opts_block, struct = cut_block(struct, "ROAD_CRG_OPTS")
    mods_block, struct = cut_block(struct, "ROAD_CRG_MODS")
    fref_block, struct = cut_block(struct, "ROAD_CRG_FILE")
    if fref_block is not None:
        ref = _resolve_file_reference(fref_block, out.warnings)
        if not os.path.isabs(ref):
            ref = os.path.join(os.path.dirname(path), ref)
        ipl = read_ipl(ref, meta_only=True)
        struct2 = ipl.struct
        o2, struct2 = cut_block(struct2, "ROAD_CRG_OPTS")
        m2, struct2 = cut_block(struct2, "ROAD_CRG_MODS")
        if opts_block is None:
            opts_block = o2
        if mods_block is None:
            mods_block = m2
        struct = struct2
        out.fmt = ipl.fmt
    if not ipl.kd_def:
        raise CrgFormatError(f"{path}: no $KD_DEFINITION block found (is this an OpenCRG file?)")

    ct, struct = cut_block(struct, "CT")
    out.comments = list(ct or [])
    all_opts = _parse_keyword_block(opts_block, set(OPTION_DEFAULTS) | set(OTHER_TOOL_OPTIONS), warns=out.warnings)
    out.opts = {k: v for k, v in all_opts.items() if k in OPTION_DEFAULTS}
    out.other_tool_opts = {k: v for k, v in all_opts.items() if k in OTHER_TOOL_OPTIONS}
    out.mods = None if mods_block is None else _parse_keyword_block(mods_block, MODIFIER_KEYWORDS, warns=out.warnings)
    head_block, struct = cut_block(struct, "ROAD_CRG")
    out.head = _parse_keyword_block(head_block, ROAD_CRG_KEYWORDS, warns=out.warnings)
    mpro_block, struct = cut_block(struct, "ROAD_CRG_MPRO")
    if mpro_block is not None:
        out.mpro = _parse_keyword_block(mpro_block, MPRO_KEYWORDS, numeric_only=False, warns=out.warnings)
    for name in _KNOWN_BLOCKS:  # a block given twice: only the first one is used (like OpenCRG does)
        while True:
            dup, rest = cut_block(struct, name)
            if dup is None:
                break
            struct = rest
            out.warnings.append(f"block ${name} appears more than once: only the first one is used, "
                                f"the others are ignored and will not be written back")
    out.extra = [s for s in struct if s.strip() and not s.startswith("* written by pyopencrg")
                 and not s.startswith("* written by crg_write") and not s.startswith("* written by ipl_write")]

    head = out.head
    # --- u axis: header has priority over the virtual U channel
    u_start, u_inc = 0.0, 0.01
    for d in ipl.kd_ind:
        items = [s.strip() for s in d.split(",")]
        if items and items[0].lower() == "reference line u" and len(items) > 1 and items[1] == "m":
            try:
                if len(items) > 2 and items[2]:
                    u_start = float(items[2])
                if len(items) > 3 and items[3]:
                    u_inc = float(items[3])
            except ValueError:
                out.warnings.append(f"cannot read virtual channel definition '{d}'")
    u_def = (u_start, u_inc) if ipl.kd_ind else None
    if "reference_line_start_u" in head:
        u_start = head["reference_line_start_u"]
    inc_from_end = "reference_line_increment" not in head and "reference_line_end_u" in head
    if "reference_line_increment" in head:
        u_inc = head["reference_line_increment"]
    if u_def is not None and (abs(u_def[0] - u_start) > 1e-3 or abs(u_def[1] - u_inc) > 1e-3 * max(u_inc, 1e-3)):
        out.warnings.append(f"u start / increment of the header ({u_start:g}, {u_inc:g}) differ from the "
                            f"virtual channel definition ({u_def[0]:g}, {u_def[1]:g}): header values used")
    if u_inc <= 0:
        out.errors.append(f"illegal increment along u: {u_inc:g}")
        u_inc = 0.01

    # --- data channels
    i_phi = i_slope = i_bank = None
    by_pos, by_num = [], []
    for i, d in enumerate(ipl.kd_def):
        items = [s.strip() for s in d.split(",")]
        name = items[0].lower()
        unit = items[1] if len(items) > 1 else ""
        if name == "reference line phi":
            if unit != "rad":
                raise CrgFormatError(f"{path}: unit of 'reference line phi' must be rad")
            i_phi = i
        elif name == "reference line slope":
            if unit != "m/m":
                raise CrgFormatError(f"{path}: unit of 'reference line slope' must be m/m")
            i_slope = i
        elif name == "reference line banking":
            if unit != "m/m":
                raise CrgFormatError(f"{path}: unit of 'reference line banking' must be m/m")
            i_bank = i
        elif name.startswith("long section"):
            if unit != "m":
                raise CrgFormatError(f"{path}: unit of '{items[0]}' must be m")
            rest = items[0][12:].strip()
            m = re.match(r"at\s+v\s*=\s*(.+)$", rest, flags=re.IGNORECASE)
            if m:
                by_pos.append((float(m.group(1)), i))
            else:
                by_num.append((float(rest), i))
        else:
            out.warnings.append(f"unknown data channel '{d}' ignored")
    if by_pos and by_num:
        raise CrgFormatError(f"{path}: long sections are defined both by position and by number")
    secs = sorted(by_pos or by_num)
    nv = len(secs)
    if nv < 2:
        raise CrgFormatError(f"{path}: at least 2 long sections are required")
    if by_pos:
        v = np.array([s[0] for s in secs], dtype=np.float64)
        for key, val in (("long_section_v_right", v[0]), ("long_section_v_left", v[-1])):
            if key in head and abs(head[key] - val) > 1e-3:
                out.warnings.append(f"header {key} = {head[key]:g} differs from the long section position {val:g}")
        if np.any(np.diff(v) <= 0):
            out.errors.append("two long sections have the same lateral position")
    else:
        nums = np.array([s[0] for s in secs])
        if np.max(np.abs(nums - np.arange(1, nv + 1))) > 1e-3:
            raise CrgFormatError(f"{path}: long section numbers must be 1, 2, ... {nv}")
        if "long_section_v_increment" in head:
            v_inc = head["long_section_v_increment"]
        elif "long_section_v_right" in head and "long_section_v_left" in head:
            v_inc = (head["long_section_v_left"] - head["long_section_v_right"]) / (nv - 1)
        else:
            v_inc = 0.01
        v_min = head.get("long_section_v_right", -v_inc * (nv - 1) / 2)
        v = v_min + np.arange(nv) * v_inc

    # --- selection, then reading of the data
    keep_v = np.arange(0, nv, sv)
    out.v = v[keep_v]
    z_cols = [secs[j][1] for j in keep_v]
    i0, i1 = 0, None
    if u is not None and not inc_from_end:
        if u[0] is not None:
            i0 = max(0, int(np.ceil((u[0] - u_start) / u_inc - 1e-9)))
        if u[1] is not None:
            i1 = int(np.floor((u[1] - u_start) / u_inc + 1e-9)) + 1
    full_cols = [c for c in (i_phi, i_slope, i_bank) if c is not None]
    try:
        z, full, nrec = read_ipl_data(ipl, slice(i0, i1, su), z_cols, full_cols, stop_on_error)
    except MemoryError:
        raise MemoryError(f"{path}: not enough memory to load this road. Read a part of it with "
                          f"crg.read(path, u=(u_first, u_last)) or a preview with crg.read(path, step=4).") from None
    out.warnings += ipl.warnings
    out.errors += ipl.errors
    if nrec < 2:
        raise CrgFormatError(f"{path}: at least 2 cross sections are required")
    if inc_from_end:
        u_inc = (head["reference_line_end_u"] - u_start) / (nrec - 1)
        if u is not None:  # the u range could not be applied before the number of records was known
            sel_u = u_start + np.arange(0, nrec, su) * u_inc
            keep = np.ones(sel_u.size, dtype=bool)
            if u[0] is not None:
                keep &= sel_u >= u[0] - 1e-9
            if u[1] is not None:
                keep &= sel_u <= u[1] + 1e-9
            z = z[keep]
            out.rows = np.arange(0, nrec, su)[keep]
    if out.rows is None:
        out.rows = np.arange(nrec)[slice(i0, i1, su)]
    if out.rows.size < 2:
        raise CrgFormatError(f"{path}: less than 2 cross sections in the requested u range")
    out.u_start, out.u_inc = float(u_start), float(u_inc)
    out.n_records = nrec
    out.z = np.ascontiguousarray(z)
    k = 0
    if i_phi is not None:
        out.phi = full[1:, k].copy()  # first value is ignored by definition
        k += 1
    if i_slope is not None:
        out.slope = full[1:, k].copy()
        k += 1
    if i_bank is not None:
        out.banking = full[:, k].copy()
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _kv(key: str, value: float, width: int = 25) -> str:
    return f"{key:<{width}} = {value:24.16e}"


def _wrap_comment(line: str) -> list:
    if len(line) <= _MAX_HEADER_LINE:
        return [line]
    out = []
    while len(line) > _MAX_HEADER_LINE:
        cut = line.rfind(" ", 0, _MAX_HEADER_LINE)
        if cut <= 0:
            cut = _MAX_HEADER_LINE
        out.append(line[:cut])
        line = line[cut:].lstrip()
    out.append(line)
    return out


def _add_block(lines: list, name: str, content: list) -> None:
    lines.append(f"${name}")
    for s in content:
        lines.append("$" + s if s.startswith("$") else s)
    lines.append("$")


def build_header(road) -> tuple:
    """Build header lines, channel definitions and data array of a road.

    Returns (struct lines, kd_ind, kd_def, data).
    """
    from . import __version__

    eps = road.get_option("check_eps")
    lines = []
    ct = []
    for c in road.comments:
        ct += _wrap_comment(c)
    if not ct:
        ct = ["OpenCRG file written by pyopencrg"]
    _add_block(lines, "CT", ct)

    opts = [_kv(k, float(v), 20) for k, v in road.opts.items()
            if k in OPTION_DEFAULTS and v != OPTION_DEFAULTS[k]]
    # options of other tools found in the source file are written back unchanged
    opts += [_kv(k, float(v), 20) for k, v in road._other_tool_opts.items()]
    if opts:
        _add_block(lines, "ROAD_CRG_OPTS", opts)

    if road.mods is not None:
        mods = [_kv(k, float(road.mods[k]), 20) for k in MODIFIER_KEYWORDS if k in road.mods]
        _add_block(lines, "ROAD_CRG_MODS", mods)

    nu = road.nu
    heads = {}
    heads["reference_line_start_u"] = road.u_start
    heads["reference_line_end_u"] = road.u_end
    heads["reference_line_increment"] = road.u_inc
    heads["long_section_v_right"] = road.v[0]
    heads["long_section_v_left"] = road.v[-1]
    if road.v_inc is not None:
        heads["long_section_v_increment"] = road.v_inc
    slope = road.slope
    banking = road.banking
    phi = road.phi
    const_slope = np.ptp(slope) <= eps
    const_bank = np.ptp(banking) <= eps
    const_phi = road.is_straight
    heads["reference_line_start_s"] = float(np.mean(slope)) if const_slope else float(slope[0])
    heads["reference_line_end_s"] = float(np.mean(slope)) if const_slope else float(slope[-1])
    heads["reference_line_start_b"] = float(np.mean(banking)) if const_bank else float(banking[0])
    heads["reference_line_end_b"] = float(np.mean(banking)) if const_bank else float(banking[-1])
    heads["reference_line_start_x"] = road.x[0]
    heads["reference_line_start_y"] = road.y[0]
    heads["reference_line_end_x"] = road.x[-1]
    heads["reference_line_end_y"] = road.y[-1]
    heads["reference_line_offset_x"] = road.x_offset
    heads["reference_line_offset_y"] = road.y_offset
    heads["reference_line_start_phi"] = road.phi_start
    heads["reference_line_end_phi"] = road.phi_end
    heads["reference_line_offset_phi"] = road.phi_offset
    heads["reference_line_start_z"] = road.zr[0]
    heads["reference_line_end_z"] = road.zr[-1]
    heads["reference_line_offset_z"] = road.z_offset
    for key, attr in (("reference_line_start_lon", "lon_start"), ("reference_line_start_lat", "lat_start"),
                      ("reference_line_end_lon", "lon_end"), ("reference_line_end_lat", "lat_end"),
                      ("reference_line_start_alt", "alt_start"), ("reference_line_end_alt", "alt_end")):
        val = getattr(road, attr)
        if val is not None:
            heads[key] = val
    _add_block(lines, "ROAD_CRG", [_kv(k, float(heads[k])) for k in ROAD_CRG_KEYWORDS if k in heads])

    if road.mpro:
        mp = []
        for k in MPRO_KEYWORDS:
            if k in road.mpro:
                val = road.mpro[k]
                mp.append(f"{k:<7} = '{val}'" if k.endswith("_nm") else _kv(k, float(val), 7))
        _add_block(lines, "ROAD_CRG_MPRO", mp)

    lines += list(road.extra_header)
    lines.append(f"* written by pyopencrg {__version__} at {datetime.now():%Y-%m-%d %H:%M:%S}")

    kd_ind = [f"reference line u,m,{road.u_start:.10g},{road.u_inc:.10g}"]
    kd_def = []
    cols = []
    if not const_phi:
        kd_def.append("reference line phi,rad")
        cols.append(np.concatenate(([np.nan], phi)))
    if not const_slope:
        kd_def.append("reference line slope,m/m")
        cols.append(np.concatenate(([np.nan], slope)))
    if not const_bank:
        kd_def.append("reference line banking,m/m")
        cols.append(banking)
    if road.v_inc is not None:
        kd_def += [f"long section {i + 1},m" for i in range(road.nv)]
    else:
        kd_def += [f"long section at v = {vv:.6f},m" for vv in road.v]
    data = np.empty((nu, len(cols) + road.nv), dtype=np.float64)
    for i, c in enumerate(cols):
        data[:, i] = c
    data[:, len(cols):] = road.z
    return lines, kd_ind, kd_def, data


def write_crg_file(road, path, fmt: str = "KRBI") -> None:
    """Write a road to an OpenCRG file. Use :meth:`pyopencrg.CrgRoad.save`."""
    struct, kd_ind, kd_def, data = build_header(road)
    for s in struct:
        if len(s) > _MAX_HEADER_LINE:
            warnings.warn(f"header line longer than {_MAX_HEADER_LINE} characters: {s[:40]}...", stacklevel=3)
    write_ipl(path, struct, kd_ind, kd_def, data, fmt)

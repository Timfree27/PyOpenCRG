"""
Keyword tables of the OpenCRG file format.

This module lists every keyword understood by pyopencrg, grouped by the
header block where it appears, together with its default value.

You normally do not need to import this module: it is used internally by the
reader, the writer and the checks. It is however a handy reference when you
want to know which options or modifiers exist::

    >>> from pyopencrg.keywords import OPTION_DEFAULTS
    >>> OPTION_DEFAULTS["border_mode_u"]
    2

Keywords are always written in lower case in pyopencrg (the file format is
case insensitive).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# $ROAD_CRG block: road parameters
# ---------------------------------------------------------------------------

#: Keywords of the ``$ROAD_CRG`` block, in the order used when writing a file.
ROAD_CRG_KEYWORDS = (
    "reference_line_start_u",
    "reference_line_end_u",
    "reference_line_increment",
    "long_section_v_right",
    "long_section_v_left",
    "long_section_v_increment",
    "reference_line_start_s",
    "reference_line_end_s",
    "reference_line_start_b",
    "reference_line_end_b",
    "reference_line_start_x",
    "reference_line_start_y",
    "reference_line_end_x",
    "reference_line_end_y",
    "reference_line_offset_x",
    "reference_line_offset_y",
    "reference_line_start_phi",
    "reference_line_end_phi",
    "reference_line_offset_phi",
    "reference_line_start_z",
    "reference_line_end_z",
    "reference_line_offset_z",
    "reference_line_start_lon",
    "reference_line_start_lat",
    "reference_line_end_lon",
    "reference_line_end_lat",
    "reference_line_start_alt",
    "reference_line_end_alt",
)

# ---------------------------------------------------------------------------
# $ROAD_CRG_OPTS block: evaluation options
# ---------------------------------------------------------------------------

#: Default value of every option of the ``$ROAD_CRG_OPTS`` block.
#:
#: Border modes (``border_mode_u`` / ``border_mode_v``) tell what happens when
#: you evaluate the road outside of its grid:
#:
#: * 0: return NaN
#: * 1: return zero (plus ``border_offset_*``)
#: * 2: keep the last value of the grid (default)
#: * 3: repeat the grid periodically
#: * 4: reflect the grid (mirror)
OPTION_DEFAULTS = {
    # elevation grid border modes in u and v directions
    "border_mode_u": 2,
    "border_mode_v": 2,
    "border_offset_u": 0.0,
    "border_offset_v": 0.0,
    "border_smooth_ubeg": 0.0,
    "border_smooth_uend": 0.0,
    # reference line continuation: 0 = straight extrapolation, 1 = close track
    "refline_continuation": 0,
    # tolerances of the checks
    "check_eps": 1.0e-6,   # relative accuracy
    "check_inc": 1.0e-3,   # minimal increment (m)
    "check_tol": 1.0e-4,   # absolute tolerance (m)
    "check_pro": 5.0e-3,   # tolerance of the map projection checks (m)
    "check_wgs": 10.0,     # tolerance of the WGS84 distance checks (m)
}

#: Options of the format that only concern other tools (messages, logging and
#: search strategy of the OpenCRG C library). pyopencrg does not use them, but
#: keeps them unchanged when a file is read and saved again.
OTHER_TOOL_OPTIONS = (
    "refline_search_far", "refline_search_close",
    "warn_msgs", "warn_curv_local", "warn_curv_global",
    "log_msgs", "log_eval", "log_eval_freq", "log_hist", "log_hist_freq", "log_stat", "log_stat_freq",
)

# ---------------------------------------------------------------------------
# $ROAD_CRG_MODS block: modifiers
# ---------------------------------------------------------------------------

#: Every modifier of the ``$ROAD_CRG_MODS`` block, in the order used when
#: writing a file. Modifiers change the road when they are applied with
#: :meth:`pyopencrg.CrgRoad.apply_mods`.
MODIFIER_KEYWORDS = (
    # scaling
    "scale_z_grid",
    "scale_slope",
    "scale_banking",
    "scale_length",
    "scale_width",
    "scale_curvature",
    # NaN handling of the elevation grid
    "grid_nan_mode",
    "grid_nan_offset",
    # re-positioning of the reference line by offset
    "refline_rotcenter_x",
    "refline_rotcenter_y",
    "refline_offset_phi",
    "refline_offset_x",
    "refline_offset_y",
    "refline_offset_z",
    # re-positioning of the reference line by reference point
    "refpoint_u",
    "refpoint_u_fraction",
    "refpoint_u_offset",
    "refpoint_v",
    "refpoint_v_fraction",
    "refpoint_v_offset",
    "refpoint_x",
    "refpoint_y",
    "refpoint_z",
    "refpoint_phi",
)

#: Modifiers that OpenCRG uses when a file has **no** ``$ROAD_CRG_MODS`` block:
#: NaN values are replaced by the last valid value of the cross section, and
#: the start of the reference line is moved to x = y = z = 0 with heading 0.
DEFAULT_MODIFIERS = {"grid_nan_mode": 2, "refpoint_phi": 0.0}

#: Modifiers that select the "by offset" re-positioning.
MODS_BY_OFFSET = ("refline_offset_phi", "refline_offset_x", "refline_offset_y", "refline_offset_z")

#: Modifiers that select the "by reference point" re-positioning.
MODS_BY_REFPOINT = (
    "refpoint_u_fraction", "refpoint_u", "refpoint_v_fraction", "refpoint_v",
    "refpoint_x", "refpoint_y", "refpoint_z", "refpoint_phi",
)

# ---------------------------------------------------------------------------
# $ROAD_CRG_MPRO block: map projection
# ---------------------------------------------------------------------------

#: Keywords of the ``$ROAD_CRG_MPRO`` block. Keywords ending with ``_nm`` are
#: names (text), all others are numbers.
MPRO_KEYWORDS = (
    "gell_nm", "gell_a", "gell_b",
    "tran_nm", "tran_ds", "tran_rx", "tran_ry", "tran_rz", "tran_tx", "tran_ty", "tran_tz",
    "lell_nm", "lell_a", "lell_b",
    "proj_nm", "proj_f0", "proj_p0", "proj_l0", "proj_e0", "proj_n0",
)

# ---------------------------------------------------------------------------
# Data channels of the $KD_DEFINITION block
# ---------------------------------------------------------------------------

#: Supported data formats. ``K`` = binary, ``L`` = text (ASCII);
#: ``R`` = single precision (float32), ``D`` = double precision (float64).
FORMATS = {
    "KRBI": "binary, single precision (float32) - smallest files",
    "KDBI": "binary, double precision (float64)",
    "LRFI": "text (ASCII), single precision, 10 characters per value",
    "LDFI": "text (ASCII), double precision, 20 characters per value",
}

#: Separator line between the header and the data.
SEPARATOR = "$" * 72

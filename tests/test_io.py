import numpy as np
import pytest

import pyopencrg as crg
from pyopencrg.io import format_values, read_ipl

FORMATS = ["KRBI", "KDBI", "LRFI", "LDFI"]
TOL = {"KRBI": 1e-6, "KDBI": 1e-12, "LRFI": 2e-7, "LDFI": 1e-12}


@pytest.mark.parametrize("n", [9, 19])
def test_format_values_width_and_precision(n):
    rng = np.random.default_rng(1)
    vals = np.concatenate([rng.normal(0, s, 200) for s in (1e-5, 1e-3, 0.1, 1, 10, 1e3, 1e6)])
    vals = np.concatenate([vals, [0.0, -0.0, 9.99999999999999, -9.9999999999, 0.999999999999, 1e30, -1e30]])
    chars = format_values(vals, n)
    assert chars.shape == (vals.size, n)
    back = np.array([float(bytes(row)) for row in chars])
    rel = np.abs(back - vals) / np.maximum(np.abs(vals), 10.0 ** (-(n - 2)))
    assert np.all(rel < (5e-4 if n == 9 else 1e-13))  # LRFI keeps 4 to 8 digits


def test_format_values_nan():
    chars = format_values([np.nan, np.inf], 9)
    assert bytes(chars[0]) == b"*" * 9 and bytes(chars[1]) == b"*" * 9


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_straight(tmp_path, straight_road, fmt):
    f = tmp_path / f"r_{fmt}.crg"
    straight_road.save(f, fmt)
    r = crg.read(f)
    assert r.shape == straight_road.shape
    assert np.allclose(r.z, straight_road.z, atol=TOL[fmt], rtol=TOL[fmt])
    assert np.allclose(r.v, straight_road.v)
    assert r.u_inc == pytest.approx(0.1)
    assert r.is_straight
    assert r.comments[0] == "test road"
    assert not r.read_warnings


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_curved(tmp_path, curved_road, fmt):
    road = curved_road.copy()
    road.z[5, 3] = np.nan
    road.mods = {"grid_nan_mode": 1.0, "refpoint_x": 5.0}
    road.set_option("border_mode_u", 1)
    road.mpro = {"proj_nm": "UTM_32U"}
    road.lat_start, road.lon_start = 48.0, 9.0
    f = tmp_path / f"c_{fmt}.crg"
    road.save(f, fmt)
    r = crg.read(f)
    tol = max(TOL[fmt], 1e-12)
    assert np.isnan(r.z[5, 3])
    m = ~np.isnan(road.z)
    assert np.allclose(r.z[m], road.z[m], atol=10 * tol)
    assert np.allclose(r.x, road.x, atol=1e-6 if fmt in ("KRBI", "LRFI") else 1e-9)
    assert np.allclose(r.y, road.y, atol=1e-6 if fmt in ("KRBI", "LRFI") else 1e-9)
    assert np.allclose(r.zr, road.zr, atol=1e-5 if fmt in ("KRBI", "LRFI") else 1e-9)
    assert np.allclose(r.banking, road.banking, atol=1e-7)
    assert r.mods == {"grid_nan_mode": 1.0, "refpoint_x": 5.0}
    assert r.opts == {"border_mode_u": 1.0}
    assert r.mpro == {"proj_nm": "UTM_32U"}
    assert r.lat_start == pytest.approx(48.0)


def test_irregular_v(tmp_path):
    v = np.array([-2.0, -1.0, -0.5, 0.0, 0.25, 2.0])
    road = crg.CrgRoad(np.random.default_rng(0).random((10, v.size)), u_inc=0.5, v=v)
    assert road.v_inc is None
    f = tmp_path / "irr.crg"
    road.save(f, "LDFI")
    r = crg.read(f)
    assert np.allclose(r.v, v)
    assert np.allclose(r.z, road.z)


def test_mods_block_states(tmp_path, straight_road):
    f = tmp_path / "m.crg"
    straight_road.save(f)
    assert crg.read(f).mods is None
    straight_road.mods = {}
    straight_road.save(f)
    assert crg.read(f).mods == {}


def test_comment_with_dollar(tmp_path, straight_road):
    straight_road.comments = ["$ price", "a " * 60]
    f = tmp_path / "c.crg"
    straight_road.save(f, "LDFI")
    r = crg.read(f)
    assert r.comments[0] == "$ price"
    assert all(len(c) <= 72 for c in r.comments)


HAND_WRITTEN = """$CT                                                 ! comment text block
CRG file written by hand
$!**********************************************************************
$ROAD_CRG                                          ! crg road parameters
REFERENCE_LINE_START_U   = 0.0
REFERENCE_LINE_END_U     = 0.02
REFERENCE_LINE_INCREMENT = 0.01
LONG_SECTION_V_RIGHT     = -0.01
LONG_SECTION_V_LEFT      = 0.01
LONG_SECTION_V_INCREMENT = 0.01
$!**********************************************************************
$ROAD_CRG_MODS
GRID_NAN_MODE       = 1     ! replace NaN with z = 0
$!**********************************************************************
$KD_Definition                                   ! data definition block
#:LDFI
D:reference line phi,rad
D:long section 1,m
D:long section 2,m
D:long section 3,m
$!**********************************************************************
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
******************* .100000000000000000 .200000000000000000 .300000000000000000
 .100000000000000000 .400000000000000000********************  .60000000000000000
 .100000000000000000 .700000000000000000 .800000000000000000 1.00000000000000000
"""


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_read_hand_written_ldfi(tmp_path, newline):
    f = tmp_path / "hand.crg"
    f.write_bytes(HAND_WRITTEN.replace("\n", newline).encode("latin-1"))
    r = crg.read(f)
    assert r.shape == (3, 3)
    assert np.allclose(r.z[0], [0.1, 0.2, 0.3])
    assert np.isnan(r.z[1, 1])
    assert r.z[2, 2] == pytest.approx(1.0)
    assert np.allclose(r.phi, 0.1)
    assert r.mods == {"grid_nan_mode": 1.0}
    filled = r.apply_mods()
    assert filled.z[1, 1] == 0.0


def test_other_tool_options_kept_but_hidden(tmp_path):
    text = HAND_WRITTEN.replace("$ROAD_CRG_MODS", "$ROAD_CRG_OPTS\nLOG_EVAL = 5\nBORDER_MODE_U = 1\n$\n$ROAD_CRG_MODS")
    f = tmp_path / "opts.crg"
    f.write_text(text)
    r = crg.read(f)
    assert r.opts == {"border_mode_u": 1.0}  # only the useful option is exposed
    assert not any("log_eval" in w for w in r.read_warnings)
    with pytest.raises(KeyError):
        r.get_option("log_eval")
    out = tmp_path / "opts2.crg"
    r.save(out, "LDFI")
    assert "log_eval" in out.read_text(encoding="latin-1")  # kept for the other tools


def test_read_ipl_meta_only(tmp_path, straight_road):
    f = tmp_path / "meta.crg"
    straight_road.save(f)
    ipl = read_ipl(f, meta_only=True)
    assert ipl.data is None
    assert ipl.fmt == "KRBI"
    assert len(ipl.kd_def) == straight_road.nv


def test_convert(tmp_path, straight_road):
    a, b = tmp_path / "a.crg", tmp_path / "b.crg"
    straight_road.save(a)
    crg.convert(a, b, "LDFI")
    assert crg.compare(crg.read(a), crg.read(b)).identical


def test_not_a_crg(tmp_path):
    f = tmp_path / "x.crg"
    f.write_text("hello\n")
    with pytest.raises(ValueError):
        crg.read(f)

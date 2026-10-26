import numpy as np
import pytest

import pyopencrg as crg
from pyopencrg import io


@pytest.mark.parametrize("fmt", ["KRBI", "KDBI", "LRFI", "LDFI"])
def test_partial_read_equals_crop(tmp_path, curved_road, fmt):
    f = tmp_path / f"p_{fmt}.crg"
    curved_road.save(f, fmt)
    full = crg.read(f)
    part = crg.read(f, u=(20.0, 35.0))
    ref = full.crop(u=(20.0, 35.0))
    assert part.shape == ref.shape
    assert part.u_start == pytest.approx(20.0)
    assert np.allclose(part.z, ref.z)
    assert np.allclose(part.x, ref.x) and np.allclose(part.y, ref.y) and np.allclose(part.zr, ref.zr)
    assert part.uv_to_z(27.3, 0.4) == pytest.approx(full.uv_to_z(27.3, 0.4), abs=1e-6)


@pytest.mark.parametrize("fmt", ["KRBI", "LDFI"])
def test_step_preview(tmp_path, curved_road, fmt):
    f = tmp_path / f"s_{fmt}.crg"
    curved_road.save(f, fmt)
    full = crg.read(f)
    prev = crg.read(f, step=4)
    assert prev.u_inc == pytest.approx(4 * full.u_inc)
    assert np.allclose(prev.z, full.z[::4, ::4])
    assert np.allclose(prev.v, full.v[::4])
    assert np.allclose(prev.x, full.x[::4]) and np.allclose(prev.y, full.y[::4])
    both = crg.read(f, u=(10, 30), step=(2, 1))
    assert both.nv == full.nv and both.u_inc == pytest.approx(2 * full.u_inc)


def test_ascii_stream_chunk_boundaries(tmp_path, curved_road):
    f = tmp_path / "chunks.crg"
    curved_road.save(f, "LDFI")
    ipl = io.read_ipl(f, meta_only=True)
    nc = len(ipl.kd_def)
    ref = io.read_ipl(f).data
    for chunk in (333, 4096, 77777):  # chunks cut records anywhere
        sel, full, n = io._ascii_stream(str(f), ipl.data_offset, nc, 20, slice(3, None, 5), list(range(nc)),
                                        [0], [], [], chunk_bytes=chunk)
        assert n == ref.shape[0]
        assert np.allclose(sel, ref[3::5], equal_nan=True)
        assert np.allclose(full[:, 0], ref[:, 0], equal_nan=True)


def test_binary_converted_to_text_is_detected(tmp_path, curved_road):
    f = tmp_path / "ok.crg"
    curved_road.save(f, "KRBI")
    raw = f.read_bytes()
    i = raw.index(b"$" * 72) + 73
    data = raw[i:]
    broken = bytearray()
    for byte in data:  # what a text editor saving as UTF-8 does to binary data
        broken += b"\xef\xbf\xbd" if byte >= 0x80 else bytes([byte])
    bad = tmp_path / "bad.crg"
    bad.write_bytes(raw[:i] + bytes(broken))
    with pytest.raises(crg.CrgCheckError, match="converted to text"):
        crg.read(bad)


def test_trailing_newline_after_binary_data(tmp_path, straight_road):
    f = tmp_path / "nl.crg"
    straight_road.save(f, "KRBI")
    with open(f, "ab") as fh:
        fh.write(b"\n")
    with pytest.warns(crg.CrgCheckWarning, match="white space"):
        r = crg.read(f)
    assert np.allclose(r.z, straight_road.z, atol=1e-6)


def test_other_virtual_channel_and_badly_written_value(tmp_path):
    from test_io import HAND_WRITTEN
    text = HAND_WRITTEN.replace("#:LDFI", "#:LDFI\nU:reference line x,m,0,1.0")
    text = text.replace("GRID_NAN_MODE       = 1 ", "GRID_NAN_MODE       = 1.000000000+e00 ")
    f = tmp_path / "virt.crg"
    f.write_text(text)
    r = crg.read(f, check=False)
    assert not any("virtual channel" in w for w in r.read_warnings)
    assert r.mods == {"grid_nan_mode": 1.0}
    assert any("badly written" in w for w in r.read_warnings)


def test_missing_referenced_file(tmp_path):
    f = tmp_path / "ref.crg"
    f.write_text("$ROAD_CRG_FILE\n../nowhere/data.crg\n$\n")
    with pytest.raises(FileNotFoundError, match="ROAD_CRG_FILE"):
        crg.read(f)


def test_duplicated_block(tmp_path, straight_road):
    f = tmp_path / "dup.crg"
    straight_road.mods = {"refpoint_x": 1.0}
    straight_road.save(f, "LDFI")
    text = f.read_text(encoding="latin-1")
    text = text.replace("$ROAD_CRG\n", "$ROAD_CRG_MODS\nrefpoint_x = 4.0\n$\n$ROAD_CRG\n", 1)
    f.write_text(text, encoding="latin-1")
    with pytest.warns(crg.CrgCheckWarning, match="more than once"):
        r = crg.read(f)
    assert r.mods == {"refpoint_x": 1.0}
    out = tmp_path / "dup2.crg"
    r.save(out, "LDFI")
    assert out.read_text(encoding="latin-1").count("$ROAD_CRG_MODS") == 1

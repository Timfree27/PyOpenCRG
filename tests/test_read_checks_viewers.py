import os
import warnings

import numpy as np
import pytest

import pyopencrg as crg


def _save(tmp_path, road, name="r.crg", fmt="KRBI"):
    f = tmp_path / name
    road.save(f, fmt)
    return f


def test_truncated_binary_is_refused(tmp_path, straight_road):
    f = _save(tmp_path, straight_road)
    data = f.read_bytes()
    f.write_bytes(data[:-1000])
    with pytest.raises(crg.CrgCheckError) as err:
        crg.read(f)
    assert "truncated" in str(err.value)
    assert "check=False" in str(err.value)
    road = crg.read(f, check=False)
    assert road.read_errors


def test_truncated_text_is_refused(tmp_path, straight_road):
    f = _save(tmp_path, straight_road, fmt="LDFI")
    lines = f.read_text().splitlines(True)
    f.write_text("".join(lines[:-3]))
    with pytest.raises(crg.CrgCheckError):
        crg.read(f)


def test_unreadable_value_is_refused(tmp_path, straight_road):
    f = _save(tmp_path, straight_road, fmt="LDFI")
    text = f.read_text()
    i = text.index("$$$$")
    i = text.index("\n", i) + 1
    f.write_text(text[:i] + " abcdefghijklmnopqrs" + text[i + 20:])
    with pytest.raises(crg.CrgCheckError):
        crg.read(f)


def test_full_nan_section_is_refused_and_repairable(tmp_path, straight_road):
    straight_road.z[0] = np.nan
    f = _save(tmp_path, straight_road)
    with pytest.raises(crg.CrgCheckError):
        crg.read(f)
    fixed = crg.read(f, check=False).trim_nan_borders()
    assert fixed.check().ok


def test_warnings_shown_once_and_strict(tmp_path, straight_road):
    straight_road.z[10, 5] = np.nan  # NaN inside a cross section: warning
    f = _save(tmp_path, straight_road)
    with pytest.warns(crg.CrgCheckWarning, match="1 warning"):
        crg.read(f)
    with pytest.raises(crg.CrgCheckError):
        crg.read(f, check="strict")


def test_clean_file_reads_silently(tmp_path, straight_road):
    f = _save(tmp_path, straight_road)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        crg.read(f)


def test_set_option_validation(straight_road):
    with pytest.raises(ValueError):
        straight_road.set_option("border_mode_u", 7)


def test_cli_check_reports_broken_file(tmp_path, straight_road, capsys):
    from pyopencrg.cli import main
    straight_road.z[0] = np.nan
    f = _save(tmp_path, straight_road)
    assert main(["check", str(f)]) == 1
    assert "ERROR" in capsys.readouterr().out
    assert main(["info", str(f)]) == 0


# --- zoomable images -------------------------------------------------------

def test_zoomable_heightmap():
    import matplotlib.pyplot as plt
    road = crg.CrgRoad(np.random.default_rng(0).random((20001, 301)), u_inc=0.01, v=1.5)
    ax = road.plot.heightmap()
    zi = ax.zoomable_image
    full_step = zi.step
    assert full_step[0] > 1
    ax.set_xlim(100.0, 101.0)
    assert zi.step == (1, 1)
    arr = zi.im.get_array()
    assert arr.shape[1] >= 100  # every grid point of the 1 m zoom is drawn
    x0, x1, _, _ = zi.im.get_extent()
    assert x0 <= 100.0 and x1 >= 101.0
    plt.close("all")


def test_irregular_v_falls_back():
    import matplotlib.pyplot as plt
    road = crg.CrgRoad(np.zeros((50, 4)), u_inc=0.1, v=[-1, 0, 0.5, 2])
    ax = road.plot.heightmap()
    assert not hasattr(ax, "zoomable_image")
    plt.close("all")


# --- pyvista viewer --------------------------------------------------------

def test_view3d_screenshot(tmp_path, curved_road):
    pytest.importorskip("pyvista")
    curved_road.z[100:110, 5:8] = np.nan
    png = tmp_path / "v.png"
    for color in ("z", "grid", "normal"):
        p = curved_road.view3d(color=color, z_scale=5, off_screen=True, show=False, screenshot=str(png))
        p.close()
    assert os.path.getsize(png) > 1000
    p = curved_road.view3d(max_points=1000, off_screen=True, show=False)
    assert p.meshes[0].n_points <= 2000
    p.close()

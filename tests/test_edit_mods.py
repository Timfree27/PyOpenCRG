import numpy as np
import pytest

import pyopencrg as crg


def test_crop_and_trim(straight_road):
    part = straight_road.crop(u=(5, 10), v=(-0.5, 0.5))
    assert part.u_start == pytest.approx(5.0)
    assert part.u_end == pytest.approx(10.0)
    assert part.v_min == pytest.approx(-0.5) and part.v_max == pytest.approx(0.5)
    assert part.uv_to_z(7.3, 0.2) == pytest.approx(straight_road.uv_to_z(7.3, 0.2))
    assert straight_road.trim(1, 2).length == pytest.approx(17.0)
    assert straight_road.nu == 201  # original unchanged


def test_crop_keeps_geometry(curved_road):
    part = curved_road.crop(u=(20, 40))
    x, y = part.uv_to_xy(30.0, 1.0)
    x0, y0 = curved_road.uv_to_xy(30.0, 1.0)
    assert (x, y) == (pytest.approx(x0), pytest.approx(y0))
    assert part.uv_to_z(30.0, 1.0) == pytest.approx(curved_road.uv_to_z(30.0, 1.0))


def test_append(curved_road):
    a = curved_road.crop(u=(0, 30))
    b = curved_road.crop(u=(30 - curved_road.u_inc, 60))
    full = crg.concatenate([a, b])
    assert full.nu == curved_road.nu
    assert np.allclose(full.x, curved_road.x, atol=1e-6)
    assert np.allclose(full.y, curved_road.y, atol=1e-6)
    assert np.allclose(full.surface_z(), curved_road.surface_z(), atol=1e-6)


def test_flip_twice(curved_road):
    back = curved_road.flip().flip()
    assert crg.compare(curved_road, back).identical
    f = curved_road.flip()
    assert f.uv_to_z(f.u_start, -1.0) == pytest.approx(curved_road.uv_to_z(curved_road.u_end, 1.0))


def test_resample(curved_road):
    r = curved_road.resample(u_inc=0.1, v_inc=0.2)
    assert r.u_inc == 0.1
    u = np.linspace(1, 59, 200)
    x1, y1 = r.uv_to_xy(u, 0.5)
    x2, y2 = curved_road.uv_to_xy(u, 0.5)
    assert np.max(np.hypot(x1 - x2, y1 - y2)) < 1e-3
    assert np.allclose(r.uv_to_z(u, 0.4), curved_road.uv_to_z(u, 0.4), atol=2e-3)


def test_banking_and_elevation_keep_surface(curved_road):
    for new in (curved_road.set_banking(0), curved_road.set_reference_elevation(),
                curved_road.set_slope(0.01)):
        assert np.allclose(new.surface_z(), curved_road.surface_z())


def test_extract_slope_banking():
    road = crg.CrgRoad(np.zeros((501, 21)), u_inc=0.1, v=1.0, slope=0.03, banking=0.02)
    merged = road.set_reference_elevation(0.0).set_banking(0.0)
    assert np.allclose(merged.slope, 0) and np.allclose(merged.banking, 0)
    s = merged.extract_slope(0.5)
    assert np.allclose(s.slope, 0.03, atol=1e-6)
    b = merged.extract_banking(0.5)
    assert np.allclose(b.banking, 0.02, atol=1e-6)
    sb = merged.separate_slope_banking(-2, -2)
    assert np.allclose(sb.slope, 0.03) and np.allclose(sb.banking, 0.02)
    assert np.allclose(sb.surface_z(), road.surface_z())


def test_fill_nan_and_holes():
    z = np.tile(np.arange(10.0), (20, 1))
    z[:, :2] = np.nan
    z[5, 5] = np.nan
    road = crg.CrgRoad(z, u_inc=0.1, v=np.arange(10) * 0.1)
    f = road.fill_nan(2)
    assert np.allclose(f.z[:, 0], 2.0) and f.z[5, 5] == pytest.approx(5.0)
    assert np.all(road.fill_nan(1, 7.0).z[:, 0] == 7.0)
    h = road.fill_holes("linear")
    assert h.z[5, 5] == pytest.approx(5.0)


def test_filter_clip_profile(straight_road):
    flat = crg.CrgRoad(np.ones((50, 20)), u_inc=0.1)
    for m in ("mean", "gauss"):
        assert np.allclose(flat.filter(m, 5).z, 1.0)
    clipped = straight_road.clip_z(0.0, 0.1)
    assert np.nanmax(clipped.surface_z()) <= 0.1 + 1e-12
    marked = straight_road.add_profile([0, 20], [1, 1], [0.4, 0.5, 0.6], [0, 0.01, 0])
    assert marked.uv_to_z(3, 0.5) - straight_road.uv_to_z(3, 0.5) == pytest.approx(0.01)
    rnd = straight_road.add_random([0, 20], [1, 1], [-2, 2], [1, 1], seed=1, amp_max=0.005, u_smooth=0.05)
    d = rnd.z - straight_road.z
    assert d.min() >= -1e-12 and d.max() <= 0.005 + 1e-12


def test_overlay(straight_road):
    bump = crg.CrgRoad(np.full((11, 31), 0.01), u_inc=0.1, v=1.5)
    o = straight_road.overlay_uv(bump, iu=(0, 10))
    assert np.allclose(o.z[:11] - straight_road.z[:11], 0.01)
    o2 = straight_road.overlay_xy(bump, outside=0.0)
    assert o2.uv_to_z(0.5, 0.0) - straight_road.uv_to_z(0.5, 0.0) == pytest.approx(0.01)
    assert o2.uv_to_z(10.0, 0.0) == pytest.approx(straight_road.uv_to_z(10.0, 0.0))


def test_ramps_and_nan_borders(straight_road):
    r = straight_road.add_ramps(1.0, 2.0)
    assert r.length == pytest.approx(23.0)
    assert r.u_start == pytest.approx(-1.0)
    z = straight_road.z.copy()
    z[:3, 4] = np.nan
    t = crg.CrgRoad(z, 0.1, straight_road.v).trim_nan_borders()
    assert t.nu == straight_road.nu - 3


def test_default_mods_move_to_origin(curved_road):
    moved = curved_road.transform(100, 50, 3, 0.7)
    back = moved.apply_mods()
    assert back.x[0] == pytest.approx(0, abs=1e-9) and back.y[0] == pytest.approx(0, abs=1e-9)
    assert back.phi_start == pytest.approx(0, abs=1e-9)
    assert back.uv_to_z(0, 0) == pytest.approx(0, abs=1e-9)
    assert back.mods == {}


def test_refpoint_mods(curved_road):
    curved_road.mods = {"refpoint_u": 20.0, "refpoint_x": 10.0, "refpoint_y": 5.0, "refpoint_phi": 1.0}
    m = curved_road.apply_mods()
    x, y = m.uv_to_xy(20.0, 0.0)
    assert (x, y) == (pytest.approx(10.0), pytest.approx(5.0))
    assert m.u_to_phi(20.0) == pytest.approx(1.0)


def test_offset_mods_and_geodetic_offsets(curved_road):
    curved_road.mods = {"refline_offset_x": 3.0, "refline_offset_phi": 0.5}
    m = curved_road.apply_mods()
    p = curved_road.uv_to_xy(15.0, 1.0)
    e1 = curved_road.xyz_to_enh(*p, 0.0)
    e2 = m.xyz_to_enh(*m.uv_to_xy(15.0, 1.0), 0.0)
    assert np.allclose(e1, e2)


def test_scale_mods(straight_road):
    straight_road.mods = {"scale_length": 2.0, "scale_z_grid": 0.5, "scale_width": 2.0}
    m = straight_road.apply_mods()
    assert m.length == pytest.approx(40.0)
    assert m.width == pytest.approx(6.0)
    assert m.uv_to_z(10.0, 0.0) == pytest.approx(0.5 * 0.01 * 5.0)


def test_compare_detects_difference(straight_road):
    other = straight_road.copy()
    other.z[10, 10] += 0.01
    res = crg.compare(straight_road, other)
    assert not res.identical
    assert res.max_diff == pytest.approx(0.01)


def test_engrave_image(tmp_path, straight_road):
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("L", (20, 10), 255)
    p = tmp_path / "logo.png"
    img.save(p)
    e = straight_road.engrave_image(p, height=0.002, width=1.0, u0=5.0, v0=0.0)
    d = e.z - straight_road.z
    assert d.max() == pytest.approx(0.002)
    assert d[: straight_road.uv_to_index(5.0)].max() == 0

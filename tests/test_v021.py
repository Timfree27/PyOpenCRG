"""Tests of the features of version 0.2.1: modifiers of parts, streaming evaluator of an open file."""

import warnings

import numpy as np
import pytest

import pyopencrg as crg

MODS = {
    "default": None,  # no modifier block: OpenCRG default modifiers
    "refpoint": {"scale_z_grid": 2.0, "scale_width": 1.5, "scale_length": 1.2, "scale_curvature": 0.8,
                 "grid_nan_mode": 2, "refpoint_u_fraction": 0.5, "refpoint_v_fraction": 0.3,
                 "refpoint_x": 100.0, "refpoint_y": -20.0, "refpoint_z": 5.0, "refpoint_phi": 1.0},
    "offset": {"grid_nan_mode": 1, "grid_nan_offset": -0.1, "refline_offset_x": 10.0, "refline_offset_y": 3.0,
               "refline_offset_z": -1.0, "refline_offset_phi": 0.5},
}


@pytest.fixture
def road_file(curved_road, tmp_path, request):
    road = curved_road.move_to(250.0, -40.0, 12.0, 2.0)
    z = road.z.copy()
    z[400:420, :5] = np.nan  # NaN values, filled by the modifiers
    road.z = z
    road.mods = MODS[request.param[0]]
    path = tmp_path / f"road.{request.param[1]}.crg"
    road.save(path, fmt=request.param[1])
    return path


@pytest.mark.parametrize("road_file", [(m, f) for m in MODS for f in ("KRBI", "LDFI")], indirect=True)
def test_parts_with_mods_are_placed_like_the_whole_road(road_file):
    full = crg.read(road_file, apply_mods=True)
    k = full.u_inc / crg.read(road_file).u_inc  # scale_length changes the u values
    with crg.open(road_file) as f:
        a = f.read(u=(20, 40), apply_mods=True)
        b = f.read(u=(30, 50), apply_mods=True)
    c = crg.read(road_file, u=(30, 50), apply_mods=True)
    rng = np.random.default_rng(0)
    u = full.u_start + rng.uniform(30, 40, 300) * k
    v = rng.uniform(full.v_min, full.v_max, 300)
    for part in (a, b, c):
        assert part.mods == {}
        assert part.u_inc == pytest.approx(full.u_inc)
        assert np.allclose(np.stack(part.uv_to_xyz(u, v)), np.stack(full.uv_to_xyz(u, v)), atol=1e-9)
        assert np.allclose(np.stack(part.uv_to_normal(u, v)), np.stack(full.uv_to_normal(u, v)), atol=1e-9)
    x, y = full.uv_to_xy(u, v)
    assert np.allclose(a.xy_to_z(x, y), b.xy_to_z(x, y), atol=1e-9)


@pytest.mark.parametrize("road_file", [("default", "KRBI")], indirect=True)
def test_step_with_mods(road_file):
    full = crg.read(road_file, apply_mods=True)
    preview = crg.read(road_file, step=4, apply_mods=True)
    assert np.allclose(preview.x, full.x[::4]) and np.allclose(preview.zr, full.zr[::4])
    keep = np.ones(preview.nu, dtype=bool)
    keep[100:105] = False  # NaN rows: filled from other neighbours on the decimated grid
    assert np.allclose(preview.z[keep], full.z[::4, ::4][keep])


@pytest.mark.parametrize("road_file", [("default", "KRBI")], indirect=True)
def test_apply_mods_on_a_part_warns(road_file):
    part = crg.read(road_file, u=(20, 40))
    with pytest.warns(UserWarning, match="part of a file"):
        part.apply_mods()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        crg.read(road_file).apply_mods()  # whole road: no warning


def test_parts_without_mods_match_the_whole_road(tmp_path):
    # a long, gentle curve: a short part alone would look straight
    road = crg.generate.synthetic_road(length=200, width=4, u_inc=0.05, v_inc=0.1, curvature=[(200, 1e-4, 1e-4)])
    road.z = 0.01 * np.sin(road.u)[:, None] * np.ones(road.nv)
    road.save(tmp_path / "r.crg")
    full = crg.read(tmp_path / "r.crg")
    with crg.open(tmp_path / "r.crg") as f:
        part = f.read(u=(100, 101))
    assert not part.is_straight
    u = np.linspace(100, 101, 77)
    v = np.linspace(-2, 2, 77)
    assert np.allclose(np.stack(part.uv_to_xy(u, v)), np.stack(full.uv_to_xy(u, v)), atol=1e-12)
    x, y = full.uv_to_xy(u, v)
    uu, vv = part.xy_to_uv(x, y)
    assert np.allclose(uu, u, atol=1e-9) and np.allclose(vv, v, atol=1e-9)


def _drive(ev_a, ev_b, full, n_steps=60, spread=1.5):
    """Same contact points on both evaluators along the road (and beyond its ends)."""
    rng = np.random.default_rng(3)
    du = rng.uniform(-spread, spread, 40)
    dv = rng.uniform(-1.0, 1.0, 40)
    way = np.linspace(full.u_start - 3, full.u_end + 3, n_steps)
    for uc in np.concatenate((way, way[::-1])):  # forward then backward
        x, y = full.uv_to_xy(uc + du, 0.8 * dv)
        za = np.stack(ev_a.xy_to_z(x, y, with_normal=True))
        zb = np.stack(ev_b.xy_to_z(x, y, with_normal=True))
        assert np.allclose(za, zb, atol=1e-9, equal_nan=True), uc
        assert np.allclose(ev_a.uv_to_gradient(uc + du, dv), ev_b.uv_to_gradient(uc + du, dv),
                           atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("road_file", [("default", "KRBI"), ("refpoint", "KRBI"), ("offset", "LDFI")],
                         indirect=True)
def test_file_evaluator_matches_the_whole_road(road_file):
    full = crg.read(road_file, apply_mods=True)
    with crg.open(road_file) as f:
        ev = f.evaluator(window=8.0, margin=2.0, apply_mods=True)
        _drive(ev, full.evaluator(), full)
        assert ev.reloads > 5
        w0, w1 = ev.window
        assert w1 - w0 < 9.0 * full.u_inc / f.u_inc
        assert ev.road.nu < full.nu / 5


@pytest.mark.parametrize("opts", [{"border_mode_u": 0}, {"border_mode_u": 1, "border_offset_u": 0.2},
                                  {"border_mode_u": 3}, {"border_mode_u": 4, "border_mode_v": 4},
                                  {"border_smooth_ubeg": 4.0, "border_smooth_uend": 3.0}])
def test_file_evaluator_borders(curved_road, tmp_path, opts):
    road = curved_road
    for k, val in opts.items():
        road.set_option(k, val)
    road.save(tmp_path / "b.crg")
    full = crg.read(tmp_path / "b.crg")
    with crg.open(tmp_path / "b.crg") as f:
        _drive(f.evaluator(window=10.0, margin=2.5), full.evaluator(), full)


def test_file_evaluator_closed_track(tmp_path):
    ring = crg.generate.synthetic_road(length=188.5, width=4, u_inc=0.05, v_inc=0.1,
                                       curvature=[(188.5, 2 * np.pi / 188.5, 2 * np.pi / 188.5)])
    ring.z = 0.01 * np.cos(3 * ring.u)[:, None] * np.ones(ring.nv)
    ring.set_option("refline_continuation", 1)
    ring.save(tmp_path / "ring.crg")
    full = crg.read(tmp_path / "ring.crg")
    with crg.open(tmp_path / "ring.crg") as f:
        _drive(f.evaluator(window=10.0, margin=2.5), full.evaluator(), full, spread=0.5)


def test_file_evaluator_arguments(tmp_path, straight_road):
    straight_road.save(tmp_path / "s.crg")
    with crg.open(tmp_path / "s.crg") as f:
        with pytest.raises(ValueError):
            f.evaluator(window=10.0, margin=5.0)
        ev = f.evaluator(window=5.0, margin=1.0, dtype="float32")
        assert ev.window is None
        assert ev.uv_to_z(10.0, 0.5) == pytest.approx(straight_road.uv_to_z(10.0, 0.5), abs=1e-6)
        assert ev.road.z.dtype == np.float32
        assert ev.uv_to_xy(10.0, 0.5) == pytest.approx(straight_road.uv_to_xy(10.0, 0.5))
    with pytest.raises(ValueError, match="closed"):
        ev.uv_to_z(18.0, 0.0)


@pytest.mark.parametrize("case", ["curved", "straight", "irregular_v", "float32"])
def test_fast_path_equals_general_path(curved_road, straight_road, case):
    from pyopencrg import evaluate as ev
    road = {"curved": curved_road, "straight": straight_road}.get(case, curved_road)
    if case == "irregular_v":
        road = road.resample(v=np.concatenate(([-2.0, -1.9, -1.5], np.linspace(-1.2, 1.2, 13), [1.7, 2.0])))
    if case == "float32":
        road = road._derive(z=road.z.astype(np.float32))
    p = ev.params(road)
    tol = 1e-7 if case == "float32" else 1e-12  # corner differences rounded in float32 (1e-9 m)
    rng = np.random.default_rng(5)
    u = rng.uniform(road.u_start, road.u_end, 500)
    v = rng.uniform(road.v_min, road.v_max, 500)
    for mode, general in ((0, lambda: ev._uv_to_z(p, u, v, False, False)),
                          (1, lambda: ev._uv_to_z(p, u, v, False, True)),
                          (2, lambda: ev._uv_to_normal(p, u, v))):
        fast = ev._fast(p, u, v, mode)
        assert fast is not None
        assert np.allclose(np.stack(fast if mode else (fast,)), np.stack(general() if mode else (general(),)),
                           rtol=0, atol=tol)
    # cells found by the x, y search give the same values
    x, y = road.uv_to_xy(u, v)
    uu, vv, cell = ev._xy_to_uv(p, x, y, None, True)
    assert (cell is None) == road.is_straight
    assert np.allclose(np.stack(ev._fast(p, uu, vv, 2, cell)), np.stack(ev._uv_to_normal(p, u, v)), atol=1e-9)
    # a point outside the grid or on a NaN cell takes the general path
    assert ev._fast(p, np.array([road.u_end + 1.0]), np.array([0.0]), 0) is None
    z = road.z.copy()
    z[10, 5] = np.nan
    q = ev.params(road._derive(z=z))
    assert ev._fast(q, np.array([road.u[10]]), np.array([road.v[5]]), 0) is None

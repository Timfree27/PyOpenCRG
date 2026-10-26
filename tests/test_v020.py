"""Tests of the features of version 0.2.0: normal, gradient, evaluator, open(), float32, optional packages."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pyopencrg as crg

# --- gradient and normal --------------------------------------------------

def test_gradient_matches_finite_differences(curved_road):
    rng = np.random.default_rng(0)
    u = rng.uniform(1, 59, 500)
    v = rng.uniform(-1.9, 1.9, 500)
    h = 1e-6
    gu, gv = curved_road.uv_to_gradient(u, v)
    fu = (curved_road.uv_to_z(u + h, v) - curved_road.uv_to_z(u - h, v)) / (2 * h)
    fv = (curved_road.uv_to_z(u, v + h) - curved_road.uv_to_z(u, v - h)) / (2 * h)
    assert np.allclose(gu, fu, atol=1e-7) and np.allclose(gv, fv, atol=1e-7)


def test_normal_matches_surface_geometry(curved_road):
    rng = np.random.default_rng(1)
    u = rng.uniform(1, 59, 500)
    v = rng.uniform(-1.9, 1.9, 500)
    h = 1e-6
    p = np.stack(curved_road.uv_to_xyz(u, v), 1)
    a = np.stack(curved_road.uv_to_xyz(u + h, v), 1) - p
    b = np.stack(curved_road.uv_to_xyz(u, v + h), 1) - p
    n = np.cross(a, b)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    nx, ny, nz = curved_road.uv_to_normal(u, v)
    assert np.allclose(np.stack((nx, ny, nz), 1), n, atol=1e-6)
    assert np.allclose(nx ** 2 + ny ** 2 + nz ** 2, 1.0)


def test_normal_of_simple_surfaces():
    flat = crg.CrgRoad(np.zeros((11, 5)), 0.1, heading=0.7)
    assert np.allclose(flat.uv_to_normal(0.5, 0.0), (0, 0, 1))
    banked = crg.CrgRoad(np.zeros((11, 5)), 0.1, banking=0.1)  # left side higher: normal leans to the right (-y)
    nx, ny, nz = banked.uv_to_normal(0.5, 0.0)
    assert nx == pytest.approx(0.0, abs=1e-12) and ny == pytest.approx(-0.1 / np.sqrt(1.01))
    sloped = crg.CrgRoad(np.zeros((11, 5)), 0.1, slope=0.05)
    assert sloped.uv_to_gradient(0.5, 0.0) == (pytest.approx(0.05), pytest.approx(0.0))


def test_with_normal_variants(curved_road):
    x, y = curved_road.uv_to_xy([10.0, 30.0], [0.5, -1.0])
    z, nx, ny, nz = curved_road.xy_to_z(x, y, with_normal=True)
    assert np.allclose(z, curved_road.xy_to_z(x, y))
    assert np.allclose(np.stack(curved_road.xy_to_normal(x, y)), np.stack((nx, ny, nz)))
    z2, *_ = curved_road.uv_to_z([10.0, 30.0], [0.5, -1.0], with_normal=True)
    assert np.allclose(z, z2)
    with pytest.raises(ValueError):
        curved_road.uv_to_z(1.0, 0.0, grid_only=True, with_normal=True)


def test_gradient_border_modes(straight_road):
    straight_road.set_option("border_mode_u", 2)
    gu, gv = straight_road.uv_to_gradient(25.0, 0.0)  # beyond the end: value kept constant along u
    assert gu == 0.0 and gv == pytest.approx(0.02)
    straight_road.set_option("border_mode_u", 4)
    gu, _ = straight_road.uv_to_gradient(25.0, 0.0)   # reflected: the slope changes sign
    assert gu == pytest.approx(-0.01)
    straight_road.set_option("border_mode_u", 0)
    assert np.isnan(straight_road.uv_to_gradient(25.0, 0.0)[0])


# --- evaluator --------------------------------------------------------------

def test_evaluator_same_results_and_warm_start(curved_road):
    ev = curved_road.evaluator()
    rng = np.random.default_rng(2)
    u = rng.uniform(5, 55, 300)
    v = rng.uniform(-1.5, 1.5, 300)
    x, y = curved_road.uv_to_xy(u, v)
    u1, v1 = ev.xy_to_uv(x, y)
    assert np.allclose(u1, u, atol=1e-9) and np.allclose(v1, v, atol=1e-9)
    # points moving a little: warm start from the previous call
    x2, y2 = curved_road.uv_to_xy(u + 0.02, v)
    u2, _ = ev.xy_to_uv(x2, y2)
    assert np.allclose(u2, u + 0.02, atol=1e-9)
    # completely different points with the same count: still correct
    x3, y3 = curved_road.uv_to_xy(60 - u, -v)
    u3, v3 = ev.xy_to_uv(x3, y3)
    assert np.allclose(u3, 60 - u, atol=1e-9) and np.allclose(v3, -v, atol=1e-9)
    z, nx, ny, nz = ev.xy_to_z(x, y, with_normal=True)
    assert np.allclose(z, curved_road.xy_to_z(x, y))
    assert np.allclose(nz, curved_road.xy_to_normal(x, y)[2])
    assert isinstance(ev.uv_to_z(10.0, 0.0), float)
    assert ev.uv_to_xy(10.0, 0.0) == curved_road.uv_to_xy(10.0, 0.0)
    ev.reset()


def test_scalar_u_hint(curved_road):
    rng = np.random.default_rng(3)
    u = 30 + rng.uniform(-0.3, 0.3, 200)
    v = rng.uniform(-1, 1, 200)
    x, y = curved_road.uv_to_xy(u, v)
    u2, v2 = curved_road.xy_to_uv(x, y, u_hint=30.0)
    assert np.allclose(u2, u, atol=1e-9) and np.allclose(v2, v, atol=1e-9)


def test_evaluation_without_scipy_search(curved_road, monkeypatch):
    from pyopencrg import evaluate
    p = evaluate.params(curved_road)
    p.cache.pop("kdtree", None)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)  # simulate a missing scipy
    u = np.linspace(1, 59, 100)
    x, y = curved_road.uv_to_xy(u, 0.7)
    u2, v2 = curved_road.xy_to_uv(x, y)
    assert np.allclose(u2, u, atol=1e-9) and np.allclose(v2, 0.7, atol=1e-9)


# --- open() and float32 -------------------------------------------------------

@pytest.mark.parametrize("fmt", ["KRBI", "LDFI"])
def test_open_reads_like_read(tmp_path, curved_road, fmt):
    f = tmp_path / f"o_{fmt}.crg"
    curved_road.save(f, fmt)
    with crg.open(f) as fh:
        assert fh.nu == curved_road.nu and fh.nv == curved_road.nv
        assert fh.u_end == pytest.approx(curved_road.u_end)
        assert np.allclose(fh.x, curved_road.x, atol=1e-6)
        part = fh.read(u=(20, 35))
        ref = crg.read(f, u=(20, 35))
        assert np.array_equal(part.z, ref.z) and np.array_equal(part.x, ref.x) and part.u_start == ref.u_start
        prev = fh.read(step=(2, 3))
        assert np.array_equal(prev.z, crg.read(f, step=(2, 3)).z)
        assert crg.compare(fh.read(), crg.read(f)).identical
        part.z += 1.0  # the parts are independent copies
        assert not np.array_equal(fh.read(u=(20, 35)).z, part.z)
        assert "open" in repr(fh)
    assert fh.closed
    with pytest.raises(ValueError):
        fh.read()


def test_open_refuses_corrupted_file(tmp_path, straight_road):
    f = tmp_path / "c.crg"
    straight_road.save(f)
    f.write_bytes(f.read_bytes()[:-1003])
    with pytest.raises(crg.CrgCheckError):
        crg.open(f)
    with crg.open(f, check=False) as fh:
        assert fh.read(u=(0, 5), check=False).nu > 2


def test_open_bad_range(tmp_path, straight_road):
    f = tmp_path / "r.crg"
    straight_road.save(f)
    with crg.open(f) as fh, pytest.raises(ValueError, match="less than 2"):
        fh.read(u=(100, 200))


def test_float32(tmp_path, curved_road):
    f = tmp_path / "f.crg"
    curved_road.save(f, "KDBI")
    r32 = crg.read(f, dtype="float32")
    assert r32.z.dtype == np.float32 and r32.x.dtype == np.float64
    r64 = crg.read(f)
    assert np.allclose(r32.z, r64.z, atol=1e-6)
    assert r32.uv_to_z(30.0, 0.5) == pytest.approx(r64.uv_to_z(30.0, 0.5), abs=1e-6)
    r32.save(tmp_path / "f2.crg")
    with crg.open(f) as fh:
        assert fh.read(u=(0, 10), dtype="float32").z.dtype == np.float32
    with pytest.raises(ValueError):
        crg.read(f, dtype="int16")


# --- optional packages -------------------------------------------------------

def test_core_works_without_scipy_and_matplotlib(tmp_path, curved_road):
    f = tmp_path / "core.crg"
    curved_road.save(f)
    code = textwrap.dedent(f"""
        import sys
        for name in ("scipy", "matplotlib", "pyvista", "PIL"):
            sys.modules[name] = None   # behave as if these packages were not installed
        import numpy as np
        import pyopencrg as crg
        road = crg.read(r"{f}")
        x, y = road.uv_to_xy(np.linspace(1, 59, 50), 0.5)
        u, v = road.xy_to_uv(x, y)
        assert np.allclose(v, 0.5)
        z, nx, ny, nz = road.evaluator().xy_to_z(x, y, with_normal=True)
        part = road.crop(u=(10, 20)).flip().set_banking(0)
        part.save(r"{tmp_path / 'core2.crg'}", "LDFI")
        with crg.open(r"{f}") as fh:
            fh.read(u=(5, 15))
        road.check()
        try:
            road.fill_holes()
        except ImportError as exc:
            assert 'pip install "pyopencrg[scipy]"' in str(exc)
        else:
            raise AssertionError("fill_holes should need scipy")
        try:
            road.show()
        except ImportError as exc:
            assert 'pyopencrg[plot]' in str(exc)
        else:
            raise AssertionError("show should need matplotlib")
        print("CORE OK")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "CORE OK" in res.stdout, res.stderr

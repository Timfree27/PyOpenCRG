import numpy as np
import pytest

import pyopencrg as crg


def test_bilinear_exact(straight_road):
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 20, 1000)
    v = rng.uniform(-1.5, 1.5, 1000)
    assert np.allclose(straight_road.uv_to_z(u, v), 0.01 * u + 0.02 * v)


def test_scalar_and_broadcast(straight_road):
    assert isinstance(straight_road.uv_to_z(1.0, 0.0), float)
    z = straight_road.uv_to_z(np.array([[1.0], [2.0]]), np.array([0.0, 0.5, 1.0]))
    assert z.shape == (2, 3)


def test_slope_and_banking():
    road = crg.CrgRoad(np.zeros((101, 11)), u_inc=0.1, v=1.0, slope=0.05, banking=0.02, z_start=3.0)
    assert road.uv_to_z(5.0, 0.5) == pytest.approx(3 + 0.25 + 0.01)


def test_nan_corner_only_matters_inside():
    z = np.zeros((3, 3))
    z[1, 1] = np.nan
    road = crg.CrgRoad(z, u_inc=1.0, v=[-1, 0, 1])
    assert np.isnan(road.uv_to_z(0.5, -0.5))
    assert road.uv_to_z(0.0, -0.5) == 0.0  # on the edge opposite to the NaN corner
    assert np.isnan(road.uv_to_z(1.0, 0.0))


@pytest.mark.parametrize("mode,expected", [(0, np.nan), (1, 0.0), (2, 0.01 * 20)])
def test_border_modes_u(straight_road, mode, expected):
    straight_road.set_option("border_mode_u", mode)
    val = straight_road.uv_to_z(25.0, 0.0)
    assert (np.isnan(val) and np.isnan(expected)) or val == pytest.approx(expected)


def test_border_repeat_and_reflect(straight_road):
    straight_road.set_option("border_mode_u", 3)
    assert straight_road.uv_to_z(25.0, 0.0) == pytest.approx(0.05)
    straight_road.set_option("border_mode_u", 4)
    assert straight_road.uv_to_z(25.0, 0.0) == pytest.approx(0.15)


def test_straight_xy_round_trip():
    road = crg.CrgRoad(np.zeros((11, 5)), u_inc=1.0, v=2.0, heading=np.pi / 6, x_start=10, y_start=-5)
    x, y = road.uv_to_xy(4.0, 1.0)
    assert x == pytest.approx(10 + 4 * np.cos(np.pi / 6) - np.sin(np.pi / 6))
    u, v = road.xy_to_uv(x, y)
    assert (u, v) == (pytest.approx(4.0), pytest.approx(1.0))


def test_circle_geometry():
    r = 40.0
    road = crg.generate.synthetic_road(length=100, width=6, u_inc=0.01, curvature=[(100, 1 / r, 1 / r)])
    c = np.array([road.x[0], road.y[0] + r])  # start heading is 0: center on the left
    u = np.linspace(1, 99, 50)
    for v in (0.0, 2.0, -3.0):
        x, y = road.uv_to_xy(u, np.full(u.size, v))
        assert np.allclose(np.hypot(x - c[0], y - c[1]), r - v, atol=2e-3)
    assert np.allclose(road.u_to_curvature(u), 1 / r, rtol=1e-3)


def test_xy_to_uv_inverse(curved_road):
    rng = np.random.default_rng(3)
    u = rng.uniform(0, 60, 5000)
    v = rng.uniform(-2, 2, 5000)
    x, y = curved_road.uv_to_xy(u, v)
    u2, v2 = curved_road.xy_to_uv(x, y)
    assert np.allclose(u2, u, atol=1e-6)
    assert np.allclose(v2, v, atol=1e-6)
    z1 = curved_road.uv_to_z(u, v)
    assert np.allclose(curved_road.xy_to_z(x, y), z1, atol=1e-6)


def test_extrapolation_beyond_ends(curved_road):
    for u in (-5.0, 70.0):
        x, y = curved_road.uv_to_xy(u, 0.5)
        u2, v2 = curved_road.xy_to_uv(x, y)
        assert u2 == pytest.approx(u, abs=1e-9)
        assert v2 == pytest.approx(0.5, abs=1e-9)


def test_u_hint(curved_road):
    x, y = curved_road.uv_to_xy(30.0, 1.0)
    u, v = curved_road.xy_to_uv(x, y, u_hint=29.0)
    assert u == pytest.approx(30.0, abs=1e-6)


def test_phi_and_index(curved_road):
    assert curved_road.u_to_phi(0.0) == pytest.approx(0.0, abs=1e-9)
    assert curved_road.u_to_phi(1e6) == pytest.approx(curved_road.phi_end)
    assert curved_road.uv_to_index(1.0, -2.0) == (20, 0)


def test_closed_track():
    road = crg.generate.synthetic_road(length=2 * np.pi * 20, width=4 * np.pi / 4, u_inc=0.01 * np.pi,
                                       curvature=[(2 * np.pi * 20, 1 / 20, 1 / 20)])
    road.set_option("refline_continuation", 1)
    a = road.uv_to_xy(5.0, 0.0)
    b = road.uv_to_xy(5.0 + road.length, 0.0)
    assert np.allclose(a, b, atol=0.05)


def test_large_evaluation_chunks(straight_road):
    n = (1 << 20) + 1234
    u = np.linspace(0, 20, n)
    z = straight_road.uv_to_z(u, np.zeros(n))
    assert z.shape == (n,)
    assert np.allclose(z, 0.01 * u)

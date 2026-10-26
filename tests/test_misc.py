import numpy as np
import pytest

import pyopencrg as crg
from pyopencrg import wgs84
from pyopencrg.mapping import Ellipsoid, Projection, tm_forward, tm_inverse
from pyopencrg.smoothing import fir_smooth, smoothing_spline


# --- geodesy -------------------------------------------------------------

def test_vincenty_reference():
    # Flinders Peak -> Buninyong (classic Vincenty test case)
    lat1, lon1 = -(37 + 57 / 60 + 3.72030 / 3600), 144 + 25 / 60 + 29.52440 / 3600
    lat2, lon2 = -(37 + 39 / 60 + 10.15610 / 3600), 143 + 55 / 60 + 35.38390 / 3600
    d, a1, _ = wgs84.distance(lat1, lon1, lat2, lon2)
    assert d == pytest.approx(54972.271, abs=1e-3)
    assert np.degrees(a1) % 360 == pytest.approx(306 + 52 / 60 + 5.37 / 3600, abs=1e-5)
    la, lo, _ = wgs84.destination(lat1, lon1, a1, d)
    assert (la, lo) == (pytest.approx(lat2, abs=1e-9), pytest.approx(lon2, abs=1e-9))


def test_utm_round_trip():
    ell = Ellipsoid.make("WGS84")
    proj = Projection.make("UTM_31U")
    lat, lon = np.radians(48.85826), np.radians(2.29450)
    e, n, _ = tm_forward(lat, lon, 0.0, ell, proj)
    assert e == pytest.approx(448252, abs=5) and n == pytest.approx(5411935, abs=5)
    la, lo, _ = tm_inverse(e, n, 0.0, ell, proj)
    assert (la, lo) == (pytest.approx(lat, abs=1e-10), pytest.approx(lon, abs=1e-10))


def test_road_llh(straight_road):
    straight_road.mpro = {"proj_nm": "UTM_32U"}
    straight_road.x_offset, straight_road.y_offset = 500000.0, 5300000.0
    lat, lon, alt = straight_road.xyz_to_llh(10.0, 1.0, 2.0)
    x, y, z = straight_road.llh_to_xyz(lat, lon, alt)
    assert (x, y, z) == (pytest.approx(10.0, abs=1e-6), pytest.approx(1.0, abs=1e-6), pytest.approx(2.0))
    assert lon == pytest.approx(9.0, abs=1e-3)
    lat0, lon0, _ = straight_road.xyz_to_llh(straight_road.x[0], straight_road.y[0], straight_road.zr[0])
    straight_road.lat_start, straight_road.lon_start = float(lat0), float(lon0)
    assert not any("WGS84" in w for w in straight_road.check().warnings)
    straight_road.lat_start += 0.001  # about 110 m away
    assert any("WGS84 start" in w for w in straight_road.check().warnings)


def test_wgs84_start_end(straight_road):
    straight_road.lat_start, straight_road.lon_start = 45.0, 5.0
    straight_road.set_wgs84_end(90.0)
    assert straight_road.lat_end == pytest.approx(45.0, abs=1e-5)
    lat, lon = straight_road.xy_to_wgs84(10.0, 0.0)
    d = wgs84.distance(45.0, 5.0, lat, lon)[0]
    assert d == pytest.approx(10.0, abs=1e-6)
    assert straight_road.check().ok
    assert "openstreetmap" in wgs84.to_url(lat, lon)


# --- smoothing -----------------------------------------------------------

def test_smoothing_spline_limits():
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.sin(x)
    line = smoothing_spline(x, y, 0.0)
    a, b = np.polyfit(x, y, 1)
    assert np.allclose(line(x), a * x + b, atol=1e-8)
    interp = smoothing_spline(x, y, 1.0)
    assert np.allclose(interp(x), y, atol=1e-8)


def test_fir_smooth_constant():
    x = np.full(100, 3.0)
    for w in ("rectangular", "triangular", "hann", "sine"):
        assert np.allclose(fir_smooth(x, 5, w, "point"), 3.0)
    ramp = np.arange(100.0)
    assert np.allclose(fir_smooth(ramp, 5, "hann", "point"), ramp)


# --- generation ----------------------------------------------------------

def test_reference_line_from_points():
    t = np.linspace(0, np.pi, 400)
    x, y = 30 * np.cos(t), 30 * np.sin(t)
    xr, yr = crg.generate.reference_line_from_points(x, y, u_inc=0.1)
    assert np.allclose(np.hypot(np.diff(xr), np.diff(yr)), 0.1, atol=1e-6)
    assert np.allclose(np.hypot(xr, yr), 30, atol=0.02)
    road = crg.generate.road_from_points(x, y, width=4, u_inc=0.1)
    assert road.check().ok


def test_generators():
    r = crg.generate.from_function(lambda u, v: 0.01 * np.sin(u), 10, 2, 0.05)
    assert r.uv_to_z(np.pi / 2, 0) == pytest.approx(0.01, abs=1e-4)
    u, z = crg.generate.trapezoid_bumps(height=0.03, lead_in=1.0, length=2.0)
    assert z.max() == pytest.approx(0.03) and z[u < 1].max() == 0
    p = crg.generate.from_longitudinal_profile(u, z, width=2.5, v_inc=0.05)
    assert p.uv_to_z(u[np.argmax(z)], 0.3) == pytest.approx(0.03, abs=1e-3)
    h = crg.generate.hermite_curve([0, 0], [10, -1], [10, 0], [10, -1], 50)
    assert np.allclose(h[0], [0, 0]) and np.allclose(h[-1], [10, -1])
    f = crg.generate.flat_road(10, 3, 0.1)
    assert f.shape == (101, 31)


# --- analysis ------------------------------------------------------------

def test_normal_map_and_artifacts(straight_road):
    flat = crg.CrgRoad(np.zeros((50, 30)), 0.01)
    assert np.allclose(flat.normal_map(), 1.0)
    z = np.zeros((50, 30))
    z[25, 15] = 1.0
    spike = crg.CrgRoad(z, 0.01)
    mask = crg.analysis.artifact_mask(spike, 0.3, 2)
    assert mask[25, 15] and mask[25, 17] and not mask[0, 0]
    cleaned = spike.remove_artifacts(0.3, 2).fill_holes("linear")
    assert np.nanmax(np.abs(cleaned.z)) < 1e-12
    regions = spike.remove_artifacts(0.3, 2).nan_regions(margin=1)
    assert len(regions) == 1


def test_find_peaks():
    z = np.zeros((60, 40))
    z[30, 20] = 0.5
    road = crg.CrgRoad(z, 0.01)
    res = road.find_peaks(threshold=0.5, size=3)
    assert len(res) >= 1 and 30 in res.iu and 20 in res.iv
    assert len(crg.CrgRoad(np.zeros((60, 40)), 0.01).find_peaks()) == 0


def test_stats(straight_road):
    s = straight_road.stats()
    assert s["nan_ratio"] == 0 and s["max"] == pytest.approx(0.2 + 0.03)


# --- checks --------------------------------------------------------------

def test_check_curvature_and_nan():
    road = crg.generate.synthetic_road(length=10, width=8, u_inc=0.01, curvature=[(10, 1 / 3, 1 / 3)])
    rep = road.check()
    assert any("curvature" in w for w in rep.warnings)
    z = np.zeros((10, 5))
    z[3, 2] = np.nan
    z[4, :] = np.nan
    rep = crg.CrgRoad(z, 0.01).check()
    assert rep.errors and rep.warnings
    assert "ERROR" in str(rep)


def test_check_increments():
    rep = crg.CrgRoad(np.zeros((5, 5)), 0.0123).check()
    assert any("multiple" in w for w in rep.warnings)


# --- export / import -----------------------------------------------------

def test_exports(tmp_path, curved_road):
    curved_road.to_obj(tmp_path / "r.obj", step=4)
    curved_road.to_stl(tmp_path / "r.stl", step=4)
    curved_road.to_csv(tmp_path / "r.csv", step=10)
    curved_road.to_trm(tmp_path / "r.trm", step=1.0, extrapolation=5.0)
    curved_road.save_heightmap(tmp_path / "h.png")
    curved_road.save_normalmap(tmp_path / "n.png")
    curved_road.save_combined_map(tmp_path / "c.png", n_slices=2)
    for f in ("r.obj", "r.stl", "r.csv", "r.trm", "r_offset.txt", "h.png", "n.png", "c0.png", "c1.png"):
        assert (tmp_path / f).stat().st_size > 0
    verts, faces = crg.importers.read_stl(tmp_path / "r.stl")
    assert faces.shape[1] == 3 and verts.shape[1] == 3
    curved_road.lat_start, curved_road.lon_start = 48.0, 2.0
    curved_road.set_wgs84_end()
    curved_road.to_html_map(tmp_path / "m.html")
    assert "leaflet" in (tmp_path / "m.html").read_text()


def test_point_cloud_import():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 20, 20000)
    y = rng.uniform(-2, 2, 20000)
    z = 0.01 * x + 0.05 * y + 1.0
    road = crg.importers.from_point_cloud(x, y, z, u_inc=0.1, width=3, zero_start=False)
    assert road.uv_to_z(10.0, 0.5) == pytest.approx(0.01 * 10 + 0.025 + 1.0, abs=2e-3) or \
        road.uv_to_z(10.0, -0.5) == pytest.approx(0.01 * 10 + 0.025 + 1.0, abs=2e-3)


# --- CLI and plots -------------------------------------------------------

def test_cli(tmp_path, straight_road, capsys):
    from pyopencrg.cli import main
    f = tmp_path / "a.crg"
    straight_road.save(f)
    assert main(["info", str(f)]) == 0
    assert "length" in capsys.readouterr().out
    assert main(["check", str(f)]) == 0
    assert main(["convert", str(f), str(tmp_path / "b.crg"), "--fmt", "LRFI"]) == 0
    assert main(["export", str(f), str(tmp_path / "a.obj"), "--step", "5", "--u", "2", "8"]) == 0


def test_plots(curved_road):
    import matplotlib.pyplot as plt
    figs = curved_road.show(block=False)
    assert len(figs) >= 4
    for name in ("heightmap", "normalmap", "normal_distribution", "cross_sections", "long_sections",
                 "grid_limits", "slope_banking", "refline_xyz"):
        getattr(curved_road.plot, name)()
    curved_road.plot.peaks(curved_road.find_peaks())
    curved_road.plot.show_refpoints([10, 20], [1, 5])
    curved_road.compare(curved_road).show(block=False)
    plt.close("all")

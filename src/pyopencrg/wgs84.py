"""
WGS84 helpers: geodesic distance, destination point, conversion of local
x/y positions to latitude / longitude, map links.

Angles given to and returned by the public functions are in **degrees**
for latitudes and longitudes, and in **radians** for azimuths (clockwise
from north), like the OpenCRG MATLAB tools.

The geodesic computations use Vincenty's formulas on the WGS84 ellipsoid
(accuracy better than a millimetre).
"""

from __future__ import annotations

import numpy as np

__all__ = ["distance", "destination", "local_xy_to_wgs84", "road_xy_to_wgs84", "set_wgs84_end", "to_url"]

_A = 6378137.0
_F = 1.0 / 298.257223563
_B = _A * (1 - _F)


def distance(lat1, lon1, lat2, lon2):
    """Geodesic distance between two WGS84 points.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float or array_like
        Latitudes and longitudes in degrees.

    Returns
    -------
    dist : float or numpy.ndarray
        Distance (m).
    azimuth1 : float or numpy.ndarray
        Direction from point 1 to point 2 at point 1 (rad, clockwise from north).
    azimuth2 : float or numpy.ndarray
        Direction of the geodesic at point 2 (rad).
    """
    p1, l1, p2, l2 = np.broadcast_arrays(*[np.radians(np.asarray(a, dtype=np.float64)) for a in (lat1, lon1, lat2, lon2)])
    scalar = p1.ndim == 0
    p1, l1, p2, l2 = (np.atleast_1d(a).astype(np.float64) for a in (p1, l1, p2, l2))
    big_l = l2 - l1
    u1 = np.arctan((1 - _F) * np.tan(p1))
    u2 = np.arctan((1 - _F) * np.tan(p2))
    su1, cu1, su2, cu2 = np.sin(u1), np.cos(u1), np.sin(u2), np.cos(u2)
    lam = big_l.copy()
    for _ in range(200):
        sl, cl = np.sin(lam), np.cos(lam)
        ss = np.sqrt((cu2 * sl) ** 2 + (cu1 * su2 - su1 * cu2 * cl) ** 2)
        cs = su1 * su2 + cu1 * cu2 * cl
        sig = np.arctan2(ss, cs)
        with np.errstate(invalid="ignore", divide="ignore"):
            sa = np.where(ss > 0, cu1 * cu2 * sl / ss, 0.0)
        c2a = 1 - sa ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            c2sm = np.where(c2a > 0, cs - 2 * su1 * su2 / c2a, 0.0)
        c = _F / 16 * c2a * (4 + _F * (4 - 3 * c2a))
        lam_new = big_l + (1 - c) * _F * sa * (sig + c * ss * (c2sm + c * cs * (-1 + 2 * c2sm ** 2)))
        done = np.all(np.abs(lam_new - lam) < 1e-13)
        lam = lam_new
        if done:
            break
    sl, cl = np.sin(lam), np.cos(lam)
    usq = c2a * (_A ** 2 - _B ** 2) / _B ** 2
    a_ = 1 + usq / 16384 * (4096 + usq * (-768 + usq * (320 - 175 * usq)))
    b_ = usq / 1024 * (256 + usq * (-128 + usq * (74 - 47 * usq)))
    dsig = b_ * ss * (c2sm + b_ / 4 * (cs * (-1 + 2 * c2sm ** 2)
                                         - b_ / 6 * c2sm * (-3 + 4 * ss ** 2) * (-3 + 4 * c2sm ** 2)))
    dist = _B * a_ * (sig - dsig)
    az1 = np.arctan2(cu2 * sl, cu1 * su2 - su1 * cu2 * cl)
    az2 = np.arctan2(cu1 * sl, -su1 * cu2 + cu1 * su2 * cl)
    same = (np.abs(p1 - p2) < 1e-15) & (np.abs(l1 - l2) < 1e-15)
    dist[same] = 0.0
    az1[same] = 0.0
    az2[same] = 0.0
    if scalar:
        return float(dist[0]), float(az1[0]), float(az2[0])
    return dist, az1, az2


def destination(lat, lon, azimuth, dist):
    """Point reached from (lat, lon) after ``dist`` metres in direction ``azimuth``.

    Parameters
    ----------
    lat, lon : float or array_like
        Start point (degrees).
    azimuth : float or array_like
        Direction (rad, clockwise from north).
    dist : float or array_like
        Distance (m).

    Returns
    -------
    lat2, lon2 : float or numpy.ndarray
        End point (degrees).
    azimuth2 : float or numpy.ndarray
        Direction of the geodesic at the end point (rad).
    """
    p1, l1, a1, s = np.broadcast_arrays(np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)),
                                        np.asarray(azimuth, float), np.asarray(dist, float))
    scalar = p1.ndim == 0
    p1, l1, a1, s = (np.atleast_1d(x).astype(np.float64) for x in (p1, l1, a1, s))
    sa1, ca1 = np.sin(a1), np.cos(a1)
    tu1 = (1 - _F) * np.tan(p1)
    cu1 = 1 / np.sqrt(1 + tu1 ** 2)
    su1 = tu1 * cu1
    sig1 = np.arctan2(tu1, ca1)
    sa = cu1 * sa1
    c2a = 1 - sa ** 2
    usq = c2a * (_A ** 2 - _B ** 2) / _B ** 2
    a_ = 1 + usq / 16384 * (4096 + usq * (-768 + usq * (320 - 175 * usq)))
    b_ = usq / 1024 * (256 + usq * (-128 + usq * (74 - 47 * usq)))
    sig = s / (_B * a_)
    for _ in range(200):
        c2sm = np.cos(2 * sig1 + sig)
        ss, cs = np.sin(sig), np.cos(sig)
        dsig = b_ * ss * (c2sm + b_ / 4 * (cs * (-1 + 2 * c2sm ** 2)
                                             - b_ / 6 * c2sm * (-3 + 4 * ss ** 2) * (-3 + 4 * c2sm ** 2)))
        sig_new = s / (_B * a_) + dsig
        done = np.all(np.abs(sig_new - sig) < 1e-13)
        sig = sig_new
        if done:
            break
    ss, cs = np.sin(sig), np.cos(sig)
    c2sm = np.cos(2 * sig1 + sig)
    tmp = su1 * ss - cu1 * cs * ca1
    p2 = np.arctan2(su1 * cs + cu1 * ss * ca1, (1 - _F) * np.sqrt(sa ** 2 + tmp ** 2))
    lam = np.arctan2(ss * sa1, cu1 * cs - su1 * ss * ca1)
    c = _F / 16 * c2a * (4 + _F * (4 - 3 * c2a))
    big_l = lam - (1 - c) * _F * sa * (sig + c * ss * (c2sm + c * cs * (-1 + 2 * c2sm ** 2)))
    l2 = np.arctan2(np.sin(l1 + big_l), np.cos(l1 + big_l))
    a2 = np.arctan2(sa, -tmp)
    lat2, lon2 = np.degrees(p2), np.degrees(l2)
    zero = s == 0
    lat2[zero], lon2[zero], a2[zero] = np.degrees(p1[zero]), np.degrees(l1[zero]), a1[zero]
    if scalar:
        return float(lat2[0]), float(lon2[0]), float(a2[0])
    return lat2, lon2, a2


def local_xy_to_wgs84(lat1, lon1, lat2, lon2, x1, y1, x2, y2, x, y, eps=1e-6, tol=1e-4, dmin=1e-3):
    """Convert local x/y positions to WGS84 using two known points.

    The local system is placed on the ellipsoid using point 1 (position) and
    the direction from point 1 to point 2.

    Returns
    -------
    lat, lon : numpy.ndarray (degrees)
    """
    d12, a12, _ = distance(lat1, lon1, lat2, lon2)
    xy12 = np.hypot(x2 - x1, y2 - y1)
    p12 = np.arctan2(y2 - y1, x2 - x1)
    if d12 < dmin or xy12 < dmin:
        raise ValueError("the two reference points are too close")
    if abs(d12 - xy12) > max(eps * (d12 + xy12) / 2, tol):
        raise ValueError(f"inconsistent distance between the reference points: WGS84 {d12:.4f} m, x/y {xy12:.4f} m")
    sc = d12 / xy12
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pi = np.arctan2(y - y1, x - x1)
    di = np.hypot(x - x1, y - y1)
    lat, lon, _ = destination(np.full(x.shape, lat1), np.full(x.shape, lon1), a12 - (pi - p12), sc * di)
    return lat, lon


def road_xy_to_wgs84(road, x, y):
    """Latitude and longitude (degrees) of local positions (x, y) of a road."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if road.mpro:
        lat, lon, _ = road.xyz_to_llh(x, y, np.zeros_like(x))
        return lat, lon
    if road.lon_start is not None and road.lon_end is not None:
        tol = max(road.get_option("check_wgs"), 1e-4)
        return local_xy_to_wgs84(road.lat_start, road.lon_start, road.lat_end, road.lon_end,
                                 road.x[0], road.y[0], road.x[-1], road.y[-1], x, y,
                                 road.get_option("check_eps"), tol)
    if road.lon_start is not None:
        d = np.hypot(x - road.x[0], y - road.y[0])
        if np.all(d <= road.get_option("check_tol")):
            return np.full(x.shape, road.lat_start), np.full(x.shape, road.lon_start)
        raise ValueError("not enough WGS84 information: only the start position is known (see set_wgs84_end)")
    raise ValueError("the road has no WGS84 information")


def set_wgs84_end(road, x_axis_azimuth=90.0):
    """Compute the WGS84 end position of a road from its start position (in place)."""
    if road.lon_start is None:
        raise ValueError("the WGS84 start position is missing")
    p = np.arctan2(road.y[-1] - road.y[0], road.x[-1] - road.x[0])
    dist = np.hypot(road.x[-1] - road.x[0], road.y[-1] - road.y[0])
    lat, lon, _ = destination(road.lat_start, road.lon_start, np.radians(x_axis_azimuth) - p, dist)
    road.lat_end, road.lon_end = lat, lon


def to_url(lat, lon, label="OpenCRG road mark", service="osm"):
    """Web link showing a point on a map.

    Parameters
    ----------
    lat, lon : float
        Position (degrees).
    label : str
        Text of the marker (Google Maps only).
    service : str
        ``"osm"`` (OpenStreetMap, default) or ``"google"``.
    """
    if service == "google":
        return f"http://maps.google.com/maps?q={lat:.6f},{lon:.6f}({label.replace(' ', '+')})"
    return f"https://www.openstreetmap.org/?mlat={lat:.7f}&mlon={lon:.7f}#map=18/{lat:.7f}/{lon:.7f}"

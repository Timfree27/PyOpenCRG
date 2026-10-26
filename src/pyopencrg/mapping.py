"""
Map projections: link between the local road coordinates and the Earth.

The ``$ROAD_CRG_MPRO`` block of a file describes how the local x/y/z
coordinates of the road are placed on the Earth:

* ``gell_*``: global ellipsoid (default WGS84);
* ``tran_*``: datum transformation (7 parameter Helmert, default none);
* ``lell_*``: local ellipsoid (default WGS84);
* ``proj_*``: map projection: ``UTM_<zone><band>`` (e.g. ``UTM_32U``),
  ``GK3_<zone>``, ``GK6_<zone>`` (Gauss-Krueger) or ``TM`` (transverse
  Mercator with ``proj_f0, p0, l0, e0, n0``).

The chain is::

    local x, y, z  --(offsets of the road)-->  projected east, north, height
                   --(projection)-->  latitude, longitude on the local ellipsoid
                   --(datum transformation)-->  latitude, longitude, altitude (WGS84)

Angles are in radians in this module, except in the road level functions
(:func:`xyz_to_llh`, :func:`llh_to_xyz`) which use degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ELLIPSOIDS", "Ellipsoid", "Transformation", "Projection", "MapProjection",
           "geod_to_ecef", "ecef_to_geod", "helmert", "tm_forward", "tm_inverse",
           "xyz_to_enh", "enh_to_xyz", "xyz_to_llh", "llh_to_xyz"]

#: Known ellipsoids: name -> (semi-major axis a, semi-minor axis b), metres.
ELLIPSOIDS = {
    "AIRY1830": (6377563.396, 6377563.396 * (1 - 1 / 299.3249646)),
    "AIRYMODIFIED1849": (6377340.189, 6377340.189 * (1 - 1 / 299.3249646)),
    "BESSEL1841": (6377397.155, 6377397.155 * (1 - 1 / 299.1528128)),
    "BESSELMODIFIED": (6377492.018, 6377492.018 * (1 - 1 / 299.1528128)),
    "BESSELNAMIBIAGLM": (6377483.865, 6377483.865 * (1 - 1 / 299.1528128)),
    "BESSELDHDN": (6377397.155, 6377397.155 * (1 - 1 / 299.1528154)),
    "CLARKE1866": (6378206.4, 6356583.8),
    "CLARKE1866AUTHALICSPHERE": (6370997.0, 6370997.0),
    "CLARKE1880IGN": (6378249.2, 6356515.0),
    "GRS1980": (6378137.0, 6378137.0 * (1 - 1 / 298.257222101)),
    "GRS1980AUTHALICSPHERE": (6371007.0, 6371007.0),
    "INTERNATIONAL1924": (6378388.0, 6378388.0 * (1 - 1 / 297.0)),
    "INTERNATIONAL1924AUTHALICSPHERE": (6371228.0, 6371228.0),
    "KRASSOWSKY1940": (6378245.0, 6378245.0 * (1 - 1 / 298.3)),
    "WGS84": (6378137.0, 6378137.0 * (1 - 1 / 298.257223563)),
}


@dataclass
class Ellipsoid:
    """Reference ellipsoid."""

    name: str = "WGS84"
    a: float = 6378137.0
    b: float = 6378137.0 * (1 - 1 / 298.257223563)

    @classmethod
    def make(cls, name=None, a=None, b=None):
        if not name:
            name = "USERDEFINED" if a is not None else "WGS84"
        name = name.upper()
        if name == "USERDEFINED":
            if a is None:
                raise ValueError("user defined ellipsoid without semi-major axis")
            return cls(name, float(a), float(a if b is None else b))
        if name not in ELLIPSOIDS:
            raise ValueError(f"unknown ellipsoid '{name}', known: {sorted(ELLIPSOIDS)}")
        return cls(name, *ELLIPSOIDS[name])

    @property
    def e2(self):
        """First eccentricity squared."""
        return 1 - self.b ** 2 / self.a ** 2


@dataclass
class Transformation:
    """Datum transformation: ``NOP`` (none) or 7 parameter Helmert (``HL7``, ``HN7``, ``HS7``)."""

    name: str = "NOP"
    ds: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    def __post_init__(self):
        self.name = (self.name or "NOP").upper()
        if self.name not in ("NOP", "HL7", "HN7", "HS7"):
            raise ValueError(f"unknown transformation '{self.name}'")


@dataclass
class Projection:
    """Transverse Mercator projection (UTM, Gauss-Krueger or general TM)."""

    name: str = "UTM_32U"
    f0: float = 0.9996
    p0: float = 0.0
    l0: float = np.radians(9.0)
    e0: float = 500000.0
    n0: float = 0.0

    @classmethod
    def make(cls, name, f0=None, p0=None, l0=None, e0=None, n0=None):
        if not name:
            raise ValueError("undefined projection name")
        parts = name.upper().split("_", 1)
        kind = parts[0]
        zone = parts[1] if len(parts) > 1 else ""
        if kind == "GK3":
            z = int(zone)
            if not 0 <= z <= 119:
                raise ValueError(f"illegal GK3 zone {zone}")
            return cls(f"GK3_{z:03d}", 1.0, 0.0, np.radians(3 * z), 1000000 * z + 500000.0, 0.0)
        if kind == "GK6":
            z = int(zone)
            if not 0 <= z <= 59:
                raise ValueError(f"illegal GK6 zone {zone}")
            return cls(f"GK6_{z:02d}", 1.0, 0.0, np.radians(6 * z - 3), 1000000 * z + 500000.0, 0.0)
        if kind == "UTM":
            if len(zone) < 2:
                raise ValueError(f"illegal UTM zone '{zone}' (expected e.g. 32U)")
            band = zone[-1]
            bands = "CCCDEFGHJKLMNPQRSTUVWXXX"
            if band not in bands:
                raise ValueError(f"illegal UTM band letter '{band}'")
            z = int(zone[:-1])
            if not 1 <= z <= 60:
                raise ValueError(f"illegal UTM zone number {z}")
            south = bands.index(band) + 1 < 13
            return cls(f"UTM_{z:02d}{band}", 0.9996, 0.0, np.radians(6 * z - 183), 500000.0,
                       10000000.0 if south else 0.0)
        if kind == "TM":
            lon0 = 0.0 if l0 is None else float(l0)
            nm = "TM"
            if zone:
                z = int(zone)
                lon0 = np.radians(z)
                nm = f"TM_{z:03d}"
            lon0 = float(np.arctan2(np.sin(lon0), np.cos(lon0)))
            return cls(nm, 1.0 if f0 is None else float(f0), 0.0 if p0 is None else float(p0), lon0,
                       0.0 if e0 is None else float(e0), 0.0 if n0 is None else float(n0))
        raise ValueError(f"unknown projection '{name}' (use UTM_xxB, GK3_xxx, GK6_xx or TM)")


@dataclass
class MapProjection:
    """Complete map projection data of a road (``$ROAD_CRG_MPRO``)."""

    gell: Ellipsoid = field(default_factory=Ellipsoid)
    tran: Transformation = field(default_factory=Transformation)
    lell: Ellipsoid = field(default_factory=Ellipsoid)
    proj: Projection = field(default_factory=Projection)

    @classmethod
    def from_keywords(cls, kw: dict) -> MapProjection:
        """Build from the keywords of a ``$ROAD_CRG_MPRO`` block (``gell_nm``, ``proj_nm``...)."""
        g = Ellipsoid.make(kw.get("gell_nm"), kw.get("gell_a"), kw.get("gell_b"))
        lo = Ellipsoid.make(kw.get("lell_nm"), kw.get("lell_a"), kw.get("lell_b"))
        t = Transformation(kw.get("tran_nm", "NOP"), *(float(kw.get(f"tran_{k}", 0.0))
                                                       for k in ("ds", "rx", "ry", "rz", "tx", "ty", "tz")))
        p = Projection.make(kw.get("proj_nm"), kw.get("proj_f0"), kw.get("proj_p0"), kw.get("proj_l0"),
                            kw.get("proj_e0"), kw.get("proj_n0"))
        return cls(g, t, lo, p)

    # -- chain ---------------------------------------------------------
    def to_local(self, lat, lon, h):
        """WGS84 (global) latitude, longitude (rad), height -> projected east, north, height."""
        if self.gell == self.lell and self.tran.name == "NOP":
            llh = (lat, lon, h)
        else:
            xyz = geod_to_ecef(lat, lon, h, self.gell)
            llh = ecef_to_geod(*helmert(*xyz, self.tran, forward=True), self.lell)
        return tm_forward(*llh, self.lell, self.proj)

    def to_global(self, e, n, h):
        """Projected east, north, height -> WGS84 (global) latitude, longitude (rad), height."""
        lat, lon, h = tm_inverse(e, n, h, self.lell, self.proj)
        if self.gell == self.lell and self.tran.name == "NOP":
            return lat, lon, h
        xyz = geod_to_ecef(lat, lon, h, self.lell)
        return ecef_to_geod(*helmert(*xyz, self.tran, forward=False), self.gell)


def geod_to_ecef(lat, lon, h, ell: Ellipsoid):
    """Geodetic latitude, longitude (rad), height -> earth centered x, y, z."""
    e2 = ell.e2
    sp = np.sin(lat)
    n = ell.a / np.sqrt(1 - e2 * sp ** 2)
    return ((n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon), (n * (1 - e2) + h) * sp)


def ecef_to_geod(x, y, z, ell: Ellipsoid):
    """Earth centered x, y, z -> geodetic latitude, longitude (rad), height."""
    a, b = ell.a, ell.b
    e2 = 1 - b ** 2 / a ** 2
    ep2 = a ** 2 / b ** 2 - 1
    f = 1 - b / a
    lam = np.arctan2(y, x)
    r = np.hypot(x, y)
    beta = np.arctan2(z, (1 - f) * r)
    for _ in range(6):
        phi = np.arctan2(z + b * ep2 * np.sin(beta) ** 3, r - a * e2 * np.cos(beta) ** 3)
        beta_new = np.arctan2((1 - f) * np.sin(phi), np.cos(phi))
        if np.max(np.abs(beta_new - beta)) < 1e-15:
            beta = beta_new
            break
        beta = beta_new
    sp, cp = np.sin(phi), np.cos(phi)
    n = a / np.sqrt(1 - e2 * sp ** 2)
    h = r * cp + (z + e2 * n * sp) * sp - n
    return phi, lam, h


def helmert(x, y, z, tran: Transformation, forward=True):
    """7 parameter Helmert transformation of earth centered coordinates."""
    if tran.name == "NOP":
        return x, y, z
    rx, ry, rz, ds = tran.rx, tran.ry, tran.rz, tran.ds
    tv = np.array([tran.tx, tran.ty, tran.tz])
    if tran.name == "HL7":
        s = np.array([[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]]).T
    elif tran.name == "HN7":
        c1, s1, c2, s2, c3, s3 = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
        s = np.array([[c2 * c3, -c2 * s3, s2],
                      [c1 * s3 + s1 * s2 * c3, c1 * c3 - s1 * s2 * s3, -s1 * c2],
                      [s1 * s3 - c1 * s2 * c3, s1 * c3 + c1 * s2 * s3, c1 * c2]]).T
    else:  # HS7
        if not forward:
            ds, tv, rx, ry, rz = -ds, -tv, -rx, -ry, -rz
            forward = True
        s = np.array([[1 + ds, -rz, ry], [rz, 1 + ds, -rx], [-ry, rx, 1 + ds]]).T
        ds = 0.0
    pts = np.vstack([np.ravel(x), np.ravel(y), np.ravel(z)])
    if forward:
        out = tv[:, None] + (1 + ds) * (s @ pts)
    else:
        out = np.linalg.solve(s, pts - tv[:, None]) / (1 + ds)
    shape = np.shape(x)
    return out[0].reshape(shape), out[1].reshape(shape), out[2].reshape(shape)


def _meridian_arc(phi, ell: Ellipsoid, proj: Projection):
    a, b = ell.a, ell.b
    n = (a - b) / (a + b)
    dp, sp = phi - proj.p0, phi + proj.p0
    return b * proj.f0 / 24 * (6 * (4 + n * (4 + 5 * n * (1 + n))) * dp
                               - 3 * n * (24 + n * (24 + 21 * n)) * np.sin(dp) * np.cos(sp)
                               + 45 * n ** 2 * (1 + n) * np.sin(2 * dp) * np.cos(2 * sp)
                               - 35 * n ** 3 * np.sin(3 * dp) * np.cos(3 * sp))


def tm_forward(lat, lon, h, ell: Ellipsoid, proj: Projection):
    """Latitude, longitude (rad) -> transverse Mercator east, north."""
    a, e2 = ell.a, ell.e2
    cp, sp = np.cos(lat), np.sin(lat)
    cp2, sp2 = cp ** 2, sp ** 2
    nu = a * proj.f0 / np.sqrt(1 - e2 * sp2)
    nr = (1 - e2 * sp2) / (1 - e2)
    eta2 = nr - 1
    dl = np.arctan2(np.sin(lon - proj.l0), np.cos(lon - proj.l0))
    dl2 = dl ** 2
    e = proj.e0 + dl * nu * cp * (1 + dl2 * ((cp2 * nr - sp2) / 6
                                              + dl2 * ((5 + 14 * eta2) * cp2 ** 2 - (18 + 58 * eta2) * sp2 * cp2 + sp2 ** 2) / 120))
    n = proj.n0 + _meridian_arc(lat, ell, proj) + dl2 * nu * sp * cp * (
        0.5 + dl2 * (((5 + 9 * eta2) * cp2 - sp2) / 24 + dl2 * (61 * cp2 ** 2 - 58 * sp2 * cp2 + sp2 ** 2) / 720))
    return e, n, h


def tm_inverse(e, n, h, ell: Ellipsoid, proj: Projection):
    """Transverse Mercator east, north -> latitude, longitude (rad)."""
    a, b = ell.a, ell.b
    nn0 = np.asarray(n, dtype=np.float64) - proj.n0
    phid = np.full(np.shape(nn0), proj.p0, dtype=np.float64)
    marc = np.zeros_like(phid)
    for _ in range(20):
        phid = phid + (nn0 - marc) / (a * proj.f0)
        marc = _meridian_arc(phid, ell, proj)
        if np.max(np.abs(nn0 - marc)) < 1e-9:
            break
    e2 = (a ** 2 - b ** 2) / a ** 2
    s2 = np.sin(phid) ** 2
    t = np.tan(phid)
    t2 = t ** 2
    nurho = (1 - e2 * s2) / (1 - e2)
    et = (np.asarray(e, dtype=np.float64) - proj.e0) * np.sqrt(1 - e2 * s2) / (a * proj.f0)
    et2 = et ** 2
    lat = phid - et2 * t * nurho * (0.5 - et2 * ((4 + nurho + t2 * (12 - 9 * nurho)) / 24
                                                  - et2 * (61 + t2 * (90 + 45 * t2)) / 720))
    lon = proj.l0 + et / np.cos(phid) * (1 - et2 * ((nurho + 2 * t2) / 6 - et2 * ((5 + t2 * (28 + 24 * t2)) / 120
                                                     - et2 * (61 + t2 * (662 + t2 * (1320 + t2 * 720))) / 5040)))
    lon = np.arctan2(np.sin(lon), np.cos(lon))
    return lat, lon, h


# ---------------------------------------------------------------------------
# road level
# ---------------------------------------------------------------------------

def xyz_to_enh(road, x, y, z):
    """Local (x, y, z) of a road -> projected (east, north, height) using the road offsets."""
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    x0, y0 = road.x[0], road.y[0]
    c, s = np.cos(road.phi_offset), np.sin(road.phi_offset)
    dx, dy = x - x0, y - y0
    return x0 + dx * c - dy * s + road.x_offset, y0 + dx * s + dy * c + road.y_offset, z + road.z_offset


def enh_to_xyz(road, e, n, h):
    """Projected (east, north, height) -> local (x, y, z) of a road."""
    e, n, h = (np.asarray(a, dtype=np.float64) for a in (e, n, h))
    x0, y0 = road.x[0], road.y[0]
    c, s = np.cos(road.phi_offset), np.sin(road.phi_offset)
    dx, dy = e - road.x_offset - x0, n - road.y_offset - y0
    return x0 + dx * c + dy * s, y0 - dx * s + dy * c, h - road.z_offset


def _projection(road):
    if not road.mpro:
        raise ValueError("the road has no map projection data (road.mpro)")
    return MapProjection.from_keywords(road.mpro)


def xyz_to_llh(road, x, y, z):
    """Local (x, y, z) -> WGS84 latitude, longitude (degrees) and altitude."""
    e, n, h = xyz_to_enh(road, x, y, z)
    lat, lon, alt = _projection(road).to_global(e, n, h)
    return np.degrees(lat), np.degrees(lon), alt


def llh_to_xyz(road, lat, lon, alt):
    """WGS84 latitude, longitude (degrees) and altitude -> local (x, y, z)."""
    e, n, h = _projection(road).to_local(np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)),
                                         np.asarray(alt, float))
    return enh_to_xyz(road, e, n, h)

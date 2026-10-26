import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import pyopencrg as crg


@pytest.fixture
def straight_road():
    """Straight road 20 m x 3 m with a known surface z = 0.01 u + 0.02 v."""
    u = np.arange(0, 20.0001, 0.1)
    v = np.arange(-1.5, 1.5001, 0.1)
    z = 0.01 * u[:, None] + 0.02 * v[None, :]
    return crg.CrgRoad(z, u_inc=0.1, v=v, comments=["test road"])


@pytest.fixture
def curved_road():
    """Left curve of radius 50 m, with slope and banking and a bumpy grid."""
    road = crg.generate.synthetic_road(
        length=60, width=4, u_inc=0.05, v_inc=0.1,
        curvature=[(10, 0, 0), (20, 0, 1 / 50), (30, 1 / 50, 1 / 50)],
        slope=[(30, 0, 0.04)], banking=[(60, 0.0, 0.03)],
    )
    uu, vv = np.meshgrid(road.u, road.v, indexing="ij")
    road.z = 0.01 * np.sin(uu) * np.cos(2 * vv)
    road.comments = ["curved test road"]
    return road

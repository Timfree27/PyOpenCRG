"""Tests of version 0.2.2: NaN filling on the NaN cells only."""

import numpy as np
import pytest

import pyopencrg as crg


def _reference_fill(z, v, offset):
    """fill_nan mode 2, one cross section at a time."""
    z = z.astype(np.float64)
    out = z.copy()
    valid_rows = [i for i in range(z.shape[0]) if not np.isnan(z[i]).all()]
    for i, row in enumerate(z):
        ok = np.flatnonzero(~np.isnan(row))
        if ok.size == 0:
            continue
        for j in np.flatnonzero(np.isnan(row)):
            if j < ok[0]:
                out[i, j] = row[ok[0]] + offset
            elif j > ok[-1]:
                out[i, j] = row[ok[-1]] + offset
            else:
                a, b = ok[ok < j][-1], ok[ok > j][0]
                out[i, j] = row[a] + (v[j] - v[a]) / (v[b] - v[a]) * (row[b] - row[a])
    for i in range(z.shape[0]):
        if np.isnan(z[i]).all():
            k = min(valid_rows, key=lambda r: (abs(r - i), -r))  # closest, the next one on a tie
            out[i] = out[k] + offset
    return out


@pytest.mark.parametrize("seed", range(20))
def test_fill_nan_matches_reference(seed):
    rng = np.random.default_rng(seed)
    nu, nv = rng.integers(2, 20), rng.integers(2, 15)
    v = np.cumsum(rng.uniform(0.05, 0.3, nv))
    z = rng.standard_normal((nu, nv))
    z[rng.random((nu, nv)) < 0.4] = np.nan
    z[rng.integers(0, nu)] = np.nan  # a cross section without any value
    z[0] = rng.standard_normal(nv)  # at least one valid cross section
    road = crg.CrgRoad(z, 0.1, v=v)
    for offset in (0.0, -0.02):
        assert np.allclose(road.fill_nan(2, offset).z, _reference_fill(z, v, offset), rtol=0, atol=1e-12)
    filled = road.fill_nan(1, 0.5).z
    assert np.array_equal(filled[~np.isnan(z)], z[~np.isnan(z)]) and np.all(filled[np.isnan(z)] == 0.5)


def test_fill_nan_keeps_the_input_and_float32():
    z = np.array([[np.nan, 1.0, np.nan, 3.0, np.nan], [0.0, 0.0, 0.0, 0.0, 0.0]])
    road = crg.CrgRoad(z, 0.1)
    filled = road.fill_nan()
    assert np.isnan(road.z[0, 0])  # the road itself is not changed
    assert np.allclose(filled.z[0], [1.0, 1.0, 2.0, 3.0, 3.0])
    road32 = road._derive(z=road.z.astype(np.float32))
    assert road32.fill_nan().z.dtype == np.float32
    with pytest.raises(ValueError):
        crg.CrgRoad(np.full((3, 3), np.nan), 0.1).fill_nan()

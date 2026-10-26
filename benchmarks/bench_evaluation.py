"""
Evaluation speed of PyOpenCRG: time of one call for 1, 640 and 10 000 points.

Run it from the root of the repository::

    python benchmarks/bench_evaluation.py

The road is synthetic (1 km, curved, 2 cm x 5 cm grid). The contact points
move a little along the road between two calls, like the tyres of a
simulated vehicle, so the evaluators use their warm start.
"""

import os
import tempfile
import timeit
import warnings

import numpy as np

import pyopencrg as crg


def make_road():
    road = crg.generate.synthetic_road(length=1000, width=4, u_inc=0.02, v_inc=0.05,
                                       curvature=[(100, 0, 0), (200, 0, 1 / 80), (300, 1 / 80, -1 / 120),
                                                  (400, -1 / 120, 0)],
                                       slope=[(1000, 0.0, 0.02)], banking=[(1000, 0.01, 0.03)])
    rng = np.random.default_rng(0)
    road.z = 0.005 * rng.standard_normal(road.shape)
    return road


def per_call(func, repeat=7):
    """Best time of one call (ms)."""
    timer = timeit.Timer(func)
    n, _ = timer.autorange()
    return min(timer.repeat(repeat, n)) / n * 1e3


def bench_size(n, road, evaluators, rng):
    """Print the time of every method for n points."""
    u0 = 500.0 + rng.uniform(-1.0, 1.0, n)
    v0 = rng.uniform(-1.8, 1.8, n)
    # positions of the points for 200 successive calls, moving forward by 1 cm per call
    xy = [road.uv_to_xy(u0 + 0.01 * k, v0) for k in range(200)]
    cases = [
        ("uv_to_z(u, v)", lambda ev, k: ev.uv_to_z(u0, v0)),
        ("uv_to_z(u, v, with_normal=True)", lambda ev, k: ev.uv_to_z(u0, v0, with_normal=True)),
        ("xy_to_uv(x, y)", lambda ev, k: ev.xy_to_uv(*xy[k])),
        ("xy_to_z(x, y)", lambda ev, k: ev.xy_to_z(*xy[k])),
        ("xy_to_z(x, y, with_normal=True)", lambda ev, k: ev.xy_to_z(*xy[k], with_normal=True)),
    ]
    for name, call in cases:
        times = []
        for ev in evaluators.values():
            step = iter(range(10**9))
            times.append(per_call(lambda ev=ev, call=call, step=step: call(ev, next(step) % 200)))
        print(f"{name:<34}{n:>8}{times[0]:>10.3f}{times[1]:>19.3f}" + "".join(f"{t:>16.3f}" for t in times[2:]))
    print()


def main():
    warnings.simplefilter("ignore")
    road = make_road()
    path = os.path.join(tempfile.mkdtemp(), "bench.crg")
    road.save(path)
    f = crg.open(path)
    evaluators = {"road": road, "road.evaluator()": road.evaluator()}
    if hasattr(f, "evaluator"):  # version 0.2.1 and later
        evaluators["f.evaluator()"] = f.evaluator(window=40.0, margin=10.0)
    rng = np.random.default_rng(1)
    print(f"PyOpenCRG {crg.__version__}, numpy {np.__version__}")
    print(f"{'method':<34}{'points':>8}{'road':>10}{'road.evaluator()':>19}{'f.evaluator()':>16}   (ms per call)")
    for n in (1, 640, 10_000):
        bench_size(n, road, evaluators, rng)
    f.close()
    os.remove(path)


if __name__ == "__main__":
    main()

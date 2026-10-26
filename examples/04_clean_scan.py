"""
Example 4 - clean a laser scanned road: remove the artifacts and fill the holes.

Usage:  python 04_clean_scan.py my_scan.crg [--interactive]
Without argument, a scan with artificial defects is created.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import pyopencrg as crg

if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    road = crg.read(sys.argv[1])
else:
    road = crg.generate.from_function(lambda u, v: 0.003 * np.sin(3 * u) * np.cos(2 * v), 30, 3, 0.005)
    rng = np.random.default_rng(0)
    for _ in range(15):  # spikes such as those created by dust or reflections
        i, j = rng.integers(0, road.nu), rng.integers(0, road.nv)
        road.z[i, j] += rng.choice([-1, 1]) * 0.05

# 1. look at the distribution of the normals to choose the tolerance
road.plot.normal_distribution()
plt.show()

# 2. clean
if "--interactive" in sys.argv:
    from pyopencrg.cleaner import clean_interactive
    cleaned = clean_interactive(road, normal_tol=0.3, radius=15)
else:
    holes = road.remove_artifacts(normal_tol=0.3, radius=5)
    print(f"{len(holes.nan_regions())} defect area(s) found")
    cleaned = holes.fill_holes("linear")

print(f"largest change: {np.nanmax(np.abs(cleaned.z - road.z)) * 1000:.1f} mm")
cleaned.plot.heightmap()
plt.show()
cleaned.save("scan_clean.crg")

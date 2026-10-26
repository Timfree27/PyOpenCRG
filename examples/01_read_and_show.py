"""
Example 1 - read a road, print information, evaluate it and show it.

Usage:  python 01_read_and_show.py [my_road.crg]
Without argument, a demo road is created first.
"""

import sys

import numpy as np

import pyopencrg as crg

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "demo_road.crg"
    demo = crg.generate.synthetic_road(
        length=200, width=6, u_inc=0.05, v_inc=0.05,
        curvature=[(50, 0, 0), (50, 0, 1 / 80), (100, 1 / 80, 1 / 80)],
        slope=[(200, 0, 0.02)],
    )
    demo = demo.add_random([0, 200], [1, 1], [-3, 3], [1, 1], seed=1, amp_max=0.01, u_smooth=0.002)
    demo.save(path)

road = crg.read(path)

# 1. information and checks
print(road)
print()
print(road.check())

# 2. evaluation: height under a wheel every centimetre
u = np.arange(road.u_start, road.u_end, 0.01)
z_left = road.uv_to_z(u, 0.8)
print(f"\nleft wheel track: min {np.nanmin(z_left):.4f} m, max {np.nanmax(z_left):.4f} m")

# 3. position of a point and back
x, y = road.uv_to_xy(100.0, 1.0)
print(f"point u=100 m, v=1 m is at x={x:.3f}, y={y:.3f}; back: u, v = {road.xy_to_uv(x, y)}")

# 4. figures
road.show()

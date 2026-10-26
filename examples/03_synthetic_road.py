"""
Example 3 - a synthetic road with lanes, markings, ruts, curbs, a sidewalk
and a rough lane (adapted from the OpenCRG demo crg_demo_gen_sl_road).
"""

import numpy as np

import pyopencrg as crg

ubeg, uend, uinc = -20.0, 750.0, 0.5
length = uend - ubeg

road = crg.generate.synthetic_road(
    length=length, width=(-7.5, 7.5), u_inc=uinc, v_inc=0.01, u_start=ubeg,
    curvature=[(120, 0, 0), (50, 0, -1 / 50), (185.5, -1 / 50, -1 / 50), (50, -1 / 50, 0),
               (65, 0, 0), (20, 0, 1 / 30), (126.5, 1 / 30, 1 / 30), (10, 1 / 30, 0)],
    slope=[(120, 0, 0), (25, 0, 0), (25, 0, 0.03)],
    banking=[(120, 0, 0), (50, 0, -0.02)],
)
whole = [ubeg, uend]
ones = [1, 1]

# road markings: left, right and dashed center line (10 m period, 4 m marking)
road = road.add_profile(whole, ones, [3.39, 3.4, 3.6, 3.61], [0, 0.003, 0.003, 0])
road = road.add_profile(whole, ones, [-3.61, -3.6, -3.4, -3.39], [0, 0.002, 0.002, 0])
u = road.u
dashes = (np.mod(u, 10) < 4).astype(float)
road = road.add_profile(u, dashes, [-0.11, -0.1, 0.1, 0.11], [0, 0.005, 0.005, 0])

# ruts in both lanes
vr = np.arange(-0.3, 0.301, 0.05)
road = road.add_profile(whole, ones, -0.75 + vr, -0.005 * np.cos(np.pi * vr / 0.6) ** 2)
road = road.add_profile(whole, ones, -2.75 + vr, -0.008 * np.cos(np.pi * vr / 0.6) ** 2)

# curbs (10 cm high) and a sidewalk between u = 400 and 470 m
road = road.add_profile(whole, ones, [6.0, 6.2, 7.5], [0, 0.1, 0.1])
road = road.add_profile(whole, ones, [-7.5, -6.2, -6.0], [0.1, 0.1, 0])
road = road.add_profile([400, 402, 460, 470], [0, 1, 1, 0], [4.0, 4.01, 6.0], [0, 0.02, 0.02])

# rough right lane
road = road.add_random(whole, ones, [-3.4, -3.3, -0.2, -0.1], [0, 1, 1, 0], seed=2010,
                       amp_min=0.0, amp_max=0.004, u_smooth=0.0005, v_smooth=0.01)

road.comments = ["synthetic road with markings, ruts, curbs and sidewalk (pyopencrg example 3)"]
print(road)
road.save("synthetic_road.crg")
road.show()

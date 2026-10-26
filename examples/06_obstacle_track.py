"""
Example 6 - test track with trapezoidal obstacles on a -10 % slope.

A smooth Hermite transition connects a flat approach to the slope, then a
series of 3 cm high obstacles (45 degree attack angle) follows.
"""

import numpy as np

import pyopencrg as crg

dl = 0.005                     # grid step (m)
approach = 20.0                # flat approach length (m)
drop = -1.0                    # height difference at the end of the approach transition (m)
slope = -0.10                  # slope of the obstacle area

# 1. obstacles, on a flat profile
u_obs, z_obs = crg.generate.trapezoid_bumps(u_inc=dl, length=20.0, bump_length=0.23, gap=0.30,
                                            height=0.03, attack_angle=45, lead_in=10.0)

# 2. approach: Hermite curve from (0, 0) to (approach, drop) ending with the slope direction
t_end = np.array([1.0, slope]) / np.hypot(1.0, slope)
curve = crg.generate.hermite_curve([0, 0], [approach, drop], [10, 0], 10 * t_end, n=int(2 * approach / dl))
u_app = np.arange(0, approach, dl)
z_app = np.interp(u_app, curve[:, 0], curve[:, 1])

# 3. full profile: approach, then obstacles on the slope
u = np.concatenate((u_app, approach + u_obs))
z = np.concatenate((z_app, drop + slope * u_obs + z_obs))

road = crg.generate.from_longitudinal_profile(u, z, width=2.5, v_inc=dl, u_inc=dl)
road.comments = ["obstacle test track: 3 cm trapezoidal obstacles on a -10 % slope"]
road = road.extract_slope(0.01)  # move the large scale shape into the reference line, obstacles stay in the grid
print(road)
road.save("obstacle_track.crg")
road.show()

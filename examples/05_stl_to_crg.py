"""
Example 5 - convert an STL mesh (e.g. a laser scan of a test track) to OpenCRG.

Usage:  python 05_stl_to_crg.py track.stl [resolution_in_m]
Without argument, a small STL file is created first.
"""

import sys

import numpy as np

import pyopencrg as crg

if len(sys.argv) > 1:
    stl, res = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
else:
    # create a demo STL: a 10 m x 3 m strip with bumps, rotated by 30 degrees
    demo = crg.generate.from_function(lambda u, v: 0.02 * np.sin(u) ** 2, 10, 3, 0.05, heading=np.pi / 6,
                                      x_start=100, y_start=50)
    demo.to_stl("demo_scan.stl")
    stl, res = "demo_scan.stl", 0.01

# straight track: the reference line is found automatically along the main direction.
# curved track: give the center line, e.g. reference_line=(x_center, y_center)
road = crg.importers.from_stl(stl, u_inc=res, extend=1.0, zero_start=True)
print(road)
road.save(stl.rsplit(".", 1)[0] + ".crg")
road.show()

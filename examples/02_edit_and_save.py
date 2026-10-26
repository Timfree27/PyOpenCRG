"""
Example 2 - edit a road: cut, ramps, re-sample, modifiers, save in several formats.
"""

import pyopencrg as crg

road = crg.generate.from_function(lambda u, v: 0.01 * (u % 5 < 0.3), length=100, width=3, u_inc=0.01)

part = road.crop(u=(20, 60))                 # keep 40 m
part = part.add_ramps(before=2, after=2)     # smooth access ramps
part = part.resample(u_inc=0.02, v_inc=0.02) # coarser grid
part = part.move_to(x=1000, y=500, z=0, phi=0.5)   # place it in the world

part.comments.append("edited with pyopencrg")
part.mods = {"grid_nan_mode": 2}             # what the simulator should do with NaN values

for fmt in ("KRBI", "LDFI"):
    part.save(f"part_{fmt}.crg", fmt=fmt)
    print(f"saved part_{fmt}.crg")

print(crg.compare(crg.read("part_KRBI.crg"), crg.read("part_LDFI.crg")))

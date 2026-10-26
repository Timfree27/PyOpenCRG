"""
Render the pictures of the README from the ASAM OpenCRG sample road "belgian_block.crg".

The sample file is not distributed with PyOpenCRG. Get it from the ASAM OpenCRG
repository (https://github.com/asam-ev/OpenCRG, folder crg-bin), then run:

    python docs/make_images.py path/to/belgian_block.crg

Needs PyVista and Pillow: pip install "pyopencrg[viewer,image]".
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pyopencrg as crg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def main(path):
    os.makedirs(OUT, exist_ok=True)
    road = crg.read(path)
    road.filename = "belgian_block.crg"  # no local path on the pictures

    # 1. 3D view (PyVista), relief exaggerated 5 times, seen from the right side of the road, close up
    zs = 5
    um = 0.5 * (road.u_start + road.u_end)
    cx, cy = road.uv_to_xy(um, 0.0)
    cz = np.nanmean(road.z) * zs
    phi = road.u_to_phi(um)
    d = np.array([np.cos(phi), np.sin(phi)])   # driving direction
    n = np.array([-np.sin(phi), np.cos(phi)])  # to the left
    eye = np.array([cx, cy]) - 0.6 * d - 3.0 * n
    p = road.view3d(color="grid", z_scale=zs, cmap="jet", off_screen=True, show=False, show_reference_line=False)
    p.remove_bounds_axes()
    p.hide_axes()
    p.remove_actor("title")
    p.remove_scalar_bar()  # the road fills the whole picture
    p.window_size = [1600, 700]
    p.camera_position = [(eye[0], eye[1], cz + 3.2), (cx, cy, cz), (0, 0, 1)]
    p.camera.zoom(1.3)
    img = p.screenshot(return_img=True)
    Image.fromarray(img).convert("RGB").save(os.path.join(OUT, "belgian_block_3d.jpg"), quality=88)
    p.close()

    # 2. height map, then a zoom showing every grid point
    fig, axs = plt.subplots(2, 1, figsize=(12, 7.5))
    road.plot.heightmap(ax=axs[0], cmap="jet")
    axs[0].set_title("height map of the whole sample (10 m x 3.4 m)")
    road.plot.heightmap(ax=axs[1], cmap="jet")
    axs[1].set_xlim(733.0, 735.0)
    axs[1].set_ylim(-0.6, 0.6)
    axs[1].set_title("zoom: every 1 cm grid point is drawn")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "belgian_block_heightmap.png"), dpi=90)
    plt.close(fig)

    # 3. cross and long sections, reference line
    fig = road.plot.show_cuts_and_limits()
    fig.savefig(os.path.join(OUT, "belgian_block_sections.png"), dpi=80)
    plt.close(fig)
    print("pictures written to", OUT)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

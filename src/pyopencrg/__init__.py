"""
pyopencrg - read, analyse, edit, show and write OpenCRG road files in pure Python.

Unofficial implementation, not affiliated with ASAM e.V.

Quick start::

    import pyopencrg as crg

    road = crg.read("road.crg")          # read a file (KRBI, KDBI, LRFI or LDFI)
    print(road)                           # summary
    print(road.check())                   # consistency checks
    z = road.uv_to_z(10.0, 0.5)           # elevation 10 m along the road, 0.5 m to the left
    x, y = road.uv_to_xy(10.0, 0.5)       # position of that point
    part = road.crop(u=(0, 50))           # first 50 m (a new road)
    part.save("part.crg", fmt="LDFI")     # write a text file
    road.show()                           # overview figures

Main entry points:

* :func:`read` and :class:`CrgRoad` - the road object and all its methods;
* :mod:`pyopencrg.generate` - synthetic roads;
* :mod:`pyopencrg.importers` - roads from point clouds and STL meshes;
* :mod:`pyopencrg.analysis` - normal map, artifacts, holes, peaks;
* :mod:`pyopencrg.export` - OBJ, STL, CSV, images, maps, Simpack trajectories;
* :mod:`pyopencrg.plot` - all figures;
* :mod:`pyopencrg.wgs84`, :mod:`pyopencrg.mapping` - geodesy.
"""

from __future__ import annotations

__version__ = "0.1.0"

import importlib

from . import analysis, checks, edit, evaluate, generate, io, keywords, mods, smoothing
from .checks import CheckReport, CrgCheckError, CrgCheckWarning
from .edit import ComparisonResult, compare, concatenate
from .road import CrgRoad, read

__all__ = ["__version__", "read", "CrgRoad", "convert", "compare", "concatenate", "CheckReport",
           "CrgCheckError", "CrgCheckWarning", "ComparisonResult", "analysis", "checks", "edit", "evaluate",
           "generate", "io", "keywords", "mods", "smoothing", "plot", "export", "importers", "wgs84", "mapping", "cleaner", "viewer"]

_LAZY = ("plot", "export", "importers", "wgs84", "mapping", "cleaner", "viewer", "cli")


def __getattr__(name):
    # modules that import matplotlib or are rarely needed are loaded on first use
    if name in _LAZY:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'pyopencrg' has no attribute '{name}'")


def convert(source, destination, fmt="KRBI", apply_mods=False, check=True):
    """Read an OpenCRG file and write it in another format.

    Parameters
    ----------
    source, destination : str
        Input and output files.
    fmt : str
        ``"KRBI"``, ``"KDBI"``, ``"LRFI"`` or ``"LDFI"``.
    apply_mods : bool
        Apply the modifiers of the file before writing.
    check : bool or "strict"
        Check the source file, see :func:`read`.

    Examples
    --------
    >>> import pyopencrg as crg
    >>> crg.convert("road.crg", "road_text.crg", fmt="LDFI")   # doctest: +SKIP
    """
    read(source, apply_mods, check).save(destination, fmt)

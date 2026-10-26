"""
Command line interface.

After installation, the ``pyopencrg`` command is available in a terminal::

    pyopencrg info  road.crg                     summary of the road
    pyopencrg check road.crg                     consistency checks
    pyopencrg show  road.crg                     overview figures
    pyopencrg view  road.crg                     interactive 3D view (pyvista)
    pyopencrg convert road.crg road_txt.crg --fmt LDFI
    pyopencrg export road.crg road.obj           (.obj .stl .csv .png .html .trm)

Options of every command:

    --apply-mods        apply the modifiers of the file first
    --u U0 U1           use only the part of the road between u = U0 and u = U1 (m)
    --step N            keep one grid point out of N (light preview of a huge file)
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .road import read


def _read(a, check=True):
    return read(a.file, a.apply_mods, check=check, u=a.u, step=a.step)


def _cmd_info(a):
    road = _read(a, check=False)
    print(road.info())
    rep = road.check()
    print(f"\ncheck: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)"
          + (" - run 'pyopencrg check' for details" if not rep.ok else ""))


def _cmd_check(a):
    rep = _read(a, check=False).check()
    print(rep)
    return 1 if rep.errors else 0


def _cmd_view(a):
    _read(a).view3d()


def _cmd_show(a):
    _read(a).show()


def _cmd_convert(a):
    _read(a).save(a.output, a.fmt)
    print(f"written: {a.output} ({a.fmt})")


def _cmd_export(a):
    road = _read(a)
    ext = os.path.splitext(a.output)[1].lower()
    if ext == ".obj":
        road.to_obj(a.output)
    elif ext == ".stl":
        road.to_stl(a.output)
    elif ext == ".csv":
        road.to_csv(a.output)
    elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        road.save_heightmap(a.output)
    elif ext in (".html", ".htm"):
        road.to_html_map(a.output)
    elif ext == ".trm":
        road.to_trm(a.output)
    else:
        print(f"unknown output type '{ext}' (use .obj .stl .csv .png .html .trm)", file=sys.stderr)
        return 2
    print(f"written: {a.output}")
    return 0


def main(argv=None):
    """Entry point of the ``pyopencrg`` command."""
    p = argparse.ArgumentParser(prog="pyopencrg", description="Read, check, show and convert OpenCRG road files.")
    p.add_argument("--version", action="version", version=f"pyopencrg {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, func, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("file", help="OpenCRG file")
        sp.add_argument("--apply-mods", action="store_true", help="apply the modifiers of the file first")
        sp.add_argument("--u", type=float, nargs=2, metavar=("U0", "U1"), default=None,
                        help="use only the part of the road between these u values (m)")
        sp.add_argument("--step", type=int, default=1, help="keep one grid point out of STEP (light preview)")
        sp.set_defaults(func=func)
        return sp

    add("info", _cmd_info, "print a summary of the road")
    add("check", _cmd_check, "check the consistency of the road")
    add("show", _cmd_show, "show the overview figures")
    add("view", _cmd_view, "interactive 3D view (needs pyvista: pip install pyopencrg[viewer])")
    sp = add("convert", _cmd_convert, "write the road in another OpenCRG format")
    sp.add_argument("output", help="file to write")
    sp.add_argument("--fmt", default="KRBI", choices=["KRBI", "KDBI", "LRFI", "LDFI"], help="data format")
    sp = add("export", _cmd_export, "export to OBJ, STL, CSV, PNG, HTML or TRM (chosen by extension)")
    sp.add_argument("output", help="file to write")
    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Import of the optional dependencies, with a clear message when one is missing."""

from __future__ import annotations

import importlib

_EXTRA = {"scipy": "scipy", "matplotlib": "plot", "pyvista": "viewer", "PIL": "image"}


def need(module: str, feature: str):
    """Import ``module`` (e.g. ``"scipy.signal"``) or explain how to install it.

    Parameters
    ----------
    module : str
        Module to import.
    feature : str
        Name of the function that needs it, for the error message.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        top = module.split(".")[0]
        extra = _EXTRA.get(top, "all")
        raise ImportError(f"{feature} needs the '{top}' package, which is not installed. Install it with:\n"
                          f'    pip install "pyopencrg[{extra}]"\n'
                          f'or install everything with: pip install "pyopencrg[all]"') from exc

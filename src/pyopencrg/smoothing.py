"""
Smoothing tools: cubic smoothing spline and symmetric FIR filter.

These are used to extract slope and banking from measured roads, to smooth
reference lines and to filter signals along u.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SmoothingSpline", "smoothing_spline", "fir_smooth"]


class SmoothingSpline:
    """Piecewise cubic function returned by :func:`smoothing_spline`.

    Call it like a function: ``f(x)``. Values outside of the data range are
    extrapolated with the first / last polynomial piece.
    """

    def __init__(self, knots, a, b, c, d):
        self.knots = knots
        self.a, self.b, self.c, self.d = a, b, c, d

    def __call__(self, x, derivative: int = 0):
        x = np.asarray(x, dtype=np.float64)
        i = np.clip(np.searchsorted(self.knots, x, side="right") - 1, 0, self.a.size - 1)
        t = x - self.knots[i]
        a, b, c, d = self.a[i], self.b[i], self.c[i], self.d[i]
        if derivative == 0:
            return ((d * t + c) * t + b) * t + a
        if derivative == 1:
            return (3 * d * t + 2 * c) * t + b
        if derivative == 2:
            return 6 * d * t + 2 * c
        raise ValueError("derivative must be 0, 1 or 2")


def smoothing_spline(x, y, p: float, weights=None) -> SmoothingSpline:
    """Cubic smoothing spline.

    Finds the cubic spline ``f`` that minimises::

        p * sum(w_i * (y_i - f(x_i))**2) + (1 - p) * integral(f''(x)**2 dx)

    Parameters
    ----------
    x : array_like
        Increasing abscissae (at least 2 values).
    y : array_like
        Values to smooth.
    p : float in [0, 1]
        Smoothing parameter. 0 gives the least squares straight line,
        1 gives the interpolating natural cubic spline.
    weights : array_like, optional
        Weight of each point (default: all 1).

    Returns
    -------
    SmoothingSpline
        Callable piecewise cubic function.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 101)
    >>> f = smoothing_spline(x, np.sin(x), 0.99)
    >>> abs(f(5.0) - np.sin(5.0)) < 0.05
    True
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        raise ValueError("at least 2 points are required")
    if not 0.0 <= p <= 1.0:
        raise ValueError("the smoothing parameter p must be in [0, 1]")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x values must be distinct")
    n = x.size
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).ravel()[order]
    if n == 2:
        slope = (y[1] - y[0]) / h[0]
        z = np.zeros(1)
        return SmoothingSpline(x[:1], y[:1], np.array([slope]), z, z)
    hi = 1.0 / h
    # Q: n x (n-2) second difference operator, R: (n-2) x (n-2) tridiagonal
    q = diags([hi[:-1], -(hi[:-1] + hi[1:]), hi[1:]], [0, -1, -2], shape=(n, n - 2), format="csc")
    r = diags([h[1:-1] / 6.0, (h[:-1] + h[1:]) / 3.0, h[1:-1] / 6.0], [-1, 0, 1], shape=(n - 2, n - 2), format="csc")
    winv = diags(1.0 / w, 0, format="csc")
    # (p R + (1 - p) Q' W^-1 Q) g = p Q' y   with g the second derivatives,
    # written with s = g / p to stay valid for p = 0
    a_mat = (p * r + (1.0 - p) * (q.T @ winv @ q)).tocsc()
    rhs = q.T @ y
    s = np.atleast_1d(spsolve(a_mat, rhs))
    a = y - (1.0 - p) * (winv @ (q @ s))
    g = np.concatenate(([0.0], p * s, [0.0]))  # second derivative at the knots
    c = g[:-1] / 2.0
    d = (g[1:] - g[:-1]) / (6.0 * h)
    b = (a[1:] - a[:-1]) / h - h * (2.0 * g[:-1] + g[1:]) / 6.0
    return SmoothingSpline(x[:-1], a[:-1], b, c, d)


def fir_smooth(x, half_width: int, window: str = "rectangular", reflect: str = "line"):
    """Smooth a signal with a symmetric moving average (FIR) filter.

    Parameters
    ----------
    x : array_like
        Signal (1D). For a 2D array, every column is filtered.
    half_width : int
        Number of values on each side of the center value (filter length
        ``2 * half_width + 1``).
    window : str
        ``"rectangular"`` (plain moving average), ``"triangular"``,
        ``"hann"`` or ``"sine"``.
    reflect : str
        How the signal is extended at both ends: ``"line"`` (mirror),
        ``"point"`` (point reflection, keeps the trend), ``"const"`` (repeat
        the end values) or ``"zero"``.

    Returns
    -------
    numpy.ndarray
        Filtered signal, same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    w = int(half_width)
    if w <= 0 or w != half_width:
        raise ValueError("half_width must be a positive integer")
    if x.ndim == 2:
        return np.column_stack([fir_smooth(x[:, i], w, window, reflect) for i in range(x.shape[1])])
    if x.size <= w:
        raise ValueError("the signal must be longer than the filter half width")
    if reflect == "line":
        xx = np.concatenate((x[w:0:-1], x, x[-2:-w - 2:-1]))
    elif reflect == "point":
        xx = np.concatenate((2 * x[0] - x[w:0:-1], x, 2 * x[-1] - x[-2:-w - 2:-1]))
    elif reflect == "const":
        xx = np.concatenate((np.full(w, x[0]), x, np.full(w, x[-1])))
    elif reflect == "zero":
        xx = np.concatenate((np.zeros(w), x, np.zeros(w)))
    else:
        raise ValueError("reflect must be 'line', 'point', 'const' or 'zero'")
    k = np.arange(w + 1)
    if window == "rectangular":
        f = np.ones(w + 1)
    elif window == "triangular":
        f = k + 1.0
    elif window == "hann":
        f = 1 + np.cos((k - w) * (np.pi / (w + 1)))
    elif window == "sine":
        f = np.sin((k + 1) * (np.pi / 2 / (w + 1)))
    else:
        raise ValueError("window must be 'rectangular', 'triangular', 'hann' or 'sine'")
    f = np.concatenate((f, f[-2::-1]))
    f /= f.sum()
    return np.convolve(xx, f, mode="valid")

"""Shared utility functions for the crack-filling coverage planners.

Provides the small numeric helpers used across the Sensor-based Complete Coverage
(SCC) and Online SCC planners:

- **Unit conversions** between inches, millimetres, and pixels (1 px = 2 mm).
  ``inpxMap`` / ``mmpxMap`` truncate toward zero; ``pxinMap`` / ``pxmmMap`` round
  half-away-from-zero to one decimal place.
- **Rounding helpers** (``mround``, ``mfix``) that implement half-away-from-zero
  rounding and truncation-toward-zero, respectively — not Python's default
  banker's rounding (round-half-to-even).
- **Distance functions** (``spdist``, ``spdist2``) and **polyline length**
  (``total_length``) for path-cost calculations.
- **Local extrema** (``islocalmin``, ``islocalmax``) with plateau-center selection
  and NaN handling.
- **Miscellaneous** helpers: ``argwhere2d`` (fast 2-D nonzero lookup), ``smooth``
  (moving-average smoothing), ``bound`` (element-wise clamp), ``addPtsLin``
  (equidistant point insertion along a polyline), and ``vertical`` (collinearity
  test).
"""

from decimal import Decimal, ROUND_HALF_UP

import numpy as np


# --------------------------------------------------------------------------- #
# Fast 2D nonzero coordinates
# --------------------------------------------------------------------------- #
def argwhere2d(a):
    """Return the (row, col) coordinates of every nonzero element in a 2-D array.

    A drop-in replacement for ``np.argwhere(a)`` that is roughly twice as fast on
    large, mostly-empty arrays (such as crack-skeleton images) by using
    ``np.flatnonzero`` and ``np.divmod`` instead of the general ``nonzero`` path.

    Parameters
    ----------
    a : array-like, shape (M, N)
        Two-dimensional input array.

    Returns
    -------
    coords : ndarray, shape (K, 2)
        Row-column indices of the K nonzero elements, in row-major order.
        Identical output to ``np.argwhere(a)``."""
    a = np.asarray(a)
    r, c = np.divmod(np.flatnonzero(a), a.shape[1])
    return np.column_stack((r, c))


# --------------------------------------------------------------------------- #
# Rounding / truncation helpers
# --------------------------------------------------------------------------- #
def mfix(x):
    """Truncate toward zero (equivalent to C ``trunc`` / dropping the fractional part).

    Parameters
    ----------
    x : scalar or array-like

    Returns
    -------
    float or ndarray
        Scalar input returns a Python float; array input returns an ndarray."""
    r = np.trunc(np.asarray(x, dtype=float))
    return r.item() if np.isscalar(x) or np.ndim(x) == 0 else r


def _round_half_away(value: float, ndigits: int) -> float:
    """Round a single float to ``ndigits`` decimal places, half-away-from-zero.

    Uses ``Decimal(str(value))`` to avoid binary-float artifacts, so the result
    matches what a human would compute (e.g. 0.15 → 0.2, 2.45 → 2.5).
    """
    if not np.isfinite(value):
        return value
    quant = Decimal(1).scaleb(-ndigits)  # 10**-ndigits
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def mround(x, ndigits: int = 0):
    """Round half-away-from-zero to ``ndigits`` decimal places.

    Unlike Python's built-in ``round``, which uses banker's rounding (round
    half-to-even), this function always rounds 0.5 away from zero — matching
    the behavior used in the coverage planners.

    Parameters
    ----------
    x : scalar or array-like
    ndigits : int, optional
        Number of decimal places (default 0 → nearest integer).

    Returns
    -------
    float or ndarray"""
    arr = np.asarray(x, dtype=float)
    out = np.vectorize(lambda v: _round_half_away(v, ndigits))(arr)
    if np.isscalar(x) or np.ndim(x) == 0:
        return float(out)
    return out


# --------------------------------------------------------------------------- #
# Unit conversions
# --------------------------------------------------------------------------- #
def inpxMap(x):
    """Convert inches to pixels, truncating toward zero (1 px = 2 mm ≈ 0.0787 in)."""
    return mfix(np.asarray(x, dtype=float) * 25.4 / 2)


def pxinMap(x):
    """Convert pixels to inches, rounded half-away-from-zero to one decimal place."""
    return mround(np.asarray(x, dtype=float) * 2 / 25.4, 1)


def mmpxMap(x):
    """Convert millimetres to pixels, truncating toward zero (1 px = 2 mm)."""
    return mfix(np.asarray(x, dtype=float) / 2)


def pxmmMap(x):
    """Convert pixels to millimetres, rounded half-away-from-zero to one decimal place."""
    return mround(np.asarray(x, dtype=float) * 2, 1)


# --------------------------------------------------------------------------- #
# Distance / geometry helpers
# --------------------------------------------------------------------------- #
def spdist(P, Ps):
    """Euclidean distance from a single point to each point in a set.

    Parameters
    ----------
    P : array-like, shape (2,) or (1, 2)
        The query point; only the first row is used if 2-D.
    Ps : array-like, shape (N, 2)
        The set of target points.

    Returns
    -------
    distances : ndarray, shape (N,)
        Distance from ``P`` to each row of ``Ps``.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Ps = np.atleast_2d(np.asarray(Ps, dtype=float))
    return np.sqrt((P[0, 0] - Ps[:, 0]) ** 2 + (P[0, 1] - Ps[:, 1]) ** 2)


def spdist2(Ps1, Ps2):
    """Row-wise Euclidean distance between two equal-length point sets.

    Parameters
    ----------
    Ps1, Ps2 : array-like, shape (N, 2)
        Two point sets of the same length.

    Returns
    -------
    distances : ndarray, shape (N,)
        ``distances[i]`` is the distance between ``Ps1[i]`` and ``Ps2[i]``.
    """
    Ps1 = np.atleast_2d(np.asarray(Ps1, dtype=float))
    Ps2 = np.atleast_2d(np.asarray(Ps2, dtype=float))
    return np.sqrt((Ps1[:, 0] - Ps2[:, 0]) ** 2 + (Ps1[:, 1] - Ps2[:, 1]) ** 2)


def total_length(Ps):
    """Total arc length of a polyline.

    Sums the Euclidean distances between consecutive vertices. Returns 0.0 for
    arrays with fewer than two rows. NaN values propagate to the total if any
    vertex coordinate is NaN.

    Parameters
    ----------
    Ps : array-like, shape (N, 2)
        Ordered sequence of (x, y) or (row, col) vertices.

    Returns
    -------
    length : float
    """
    Ps = np.asarray(Ps, dtype=float)
    if Ps.ndim != 2 or Ps.shape[0] < 2:
        return 0.0
    d = np.diff(Ps, axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def vertical(P):
    """Return True if all points in ``P`` share the same x-coordinate (vertical line).

    Parameters
    ----------
    P : array-like, shape (N, 2)

    Returns
    -------
    bool
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return bool(np.all(P[0, 0] == P[:, 0]))


def bound(x, bl, bu):
    """Clamp ``x`` element-wise to the interval ``[bl, bu]``.

    Parameters
    ----------
    x : scalar or array-like
    bl : scalar or array-like
        Lower bound.
    bu : scalar or array-like
        Upper bound.

    Returns
    -------
    ndarray or scalar
    """
    return np.minimum(np.maximum(x, bl), bu)


# --------------------------------------------------------------------------- #
# addPtsLin: insert equidistant points along a polyline
# --------------------------------------------------------------------------- #
def addPtsLin(x, y, marker_dist):
    """Insert equidistant points along a 2-D polyline.

    Walks the polyline defined by arrays ``x`` and ``y`` and places a new point
    every ``marker_dist`` units of arc length, starting from the first vertex.
    The first and last vertices of the original polyline are not included in the
    output; only the inserted intermediate markers are returned.

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Vertex coordinates of the input polyline.
    marker_dist : float
        Spacing between successive inserted points (same units as ``x``/``y``).

    Returns
    -------
    marker_x, marker_y : ndarray, shape (K,)
        Coordinates of the K inserted points. Both arrays are empty when the
        total polyline length is less than ``marker_dist``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    dist_from_start = np.concatenate(([0.0], np.cumsum(seg)))
    total = dist_from_start[-1]

    marker_x = np.array([], dtype=float)
    marker_y = np.array([], dtype=float)
    if total < marker_dist:
        return marker_x, marker_y

    n_locs = int(np.floor(total / marker_dist + 1e-12))  # number of insertion points
    marker_locs = marker_dist * np.arange(1, n_locs + 1)

    # skip degenerate case where the only location would be the first step itself
    if marker_locs.size == 0 or (marker_locs.size == 1 and marker_locs[0] == marker_dist):
        return marker_x, marker_y

    n = dist_from_start.size
    idx_1based = np.interp(marker_locs, dist_from_start, np.arange(1, n + 1))
    base = np.floor(idx_1based).astype(int)            # 1-based base index
    w = idx_1based - base
    # 0-based: x(base) -> x[base-1], x(base+1) -> x[base]
    marker_x = x[base - 1] * (1 - w) + x[base] * w
    marker_y = y[base - 1] * (1 - w) + y[base] * w
    return marker_x, marker_y


# --------------------------------------------------------------------------- #
# islocalmin / islocalmax  (default flat_selection='center')
# --------------------------------------------------------------------------- #
def _islocal_1d(v: np.ndarray, find_min: bool) -> np.ndarray:
    """Detect local extrema in a 1-D sequence with plateau-center selection.

    Rules applied:
    - NaN values are skipped; a flat run may span across them.
    - The first and last elements are never marked as extrema.
    - For flat plateaus, the earlier middle index (floor of the run center) is marked.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    tf = np.zeros(n, dtype=bool)

    valid = np.flatnonzero(~np.isnan(v))
    m = valid.size
    if m < 3:
        return tf
    c = v[valid]  # compacted (NaN-free) values

    i = 0
    while i < m:
        j = i
        while j + 1 < m and c[j + 1] == c[i]:
            j += 1
        # run spans compacted indices [i, j]; interior only
        if i > 0 and j < m - 1:
            left, right, val = c[i - 1], c[j + 1], c[i]
            is_ext = (left > val and right > val) if find_min else (left < val and right < val)
            if is_ext:
                center = i + (j - i) // 2          # earlier-middle
                tf[valid[center]] = True
        i = j + 1
    return tf


def _islocal(A, find_min: bool) -> np.ndarray:
    """Apply ``_islocal_1d`` column-wise for 2-D arrays, or directly for 1-D."""
    A = np.asarray(A, dtype=float)
    if A.ndim <= 1:
        return _islocal_1d(A, find_min).reshape(A.shape)
    if A.ndim == 2:
        out = np.zeros(A.shape, dtype=bool)
        for k in range(A.shape[1]):
            out[:, k] = _islocal_1d(A[:, k], find_min)
        return out
    raise ValueError("islocalmin/max supports 1-D or 2-D input only")


def islocalmin(A, flat_selection: str = "center"):
    """Boolean mask of local minima in ``A``.

    For flat plateaus, the earlier middle element is selected. Endpoints and
    NaN values are never marked. Only ``flat_selection='center'`` is supported.

    Parameters
    ----------
    A : array-like, 1-D or 2-D
        Input sequence or column-oriented matrix.

    Returns
    -------
    tf : ndarray of bool, same shape as ``A``
    """
    if flat_selection != "center":
        raise NotImplementedError("only flat_selection='center' is supported")
    return _islocal(A, find_min=True)


def islocalmax(A, flat_selection: str = "center"):
    """Boolean mask of local maxima in ``A``.

    For flat plateaus, the earlier middle element is selected. Endpoints and
    NaN values are never marked. Only ``flat_selection='center'`` is supported.

    Parameters
    ----------
    A : array-like, 1-D or 2-D
        Input sequence or column-oriented matrix.

    Returns
    -------
    tf : ndarray of bool, same shape as ``A``
    """
    if flat_selection != "center":
        raise NotImplementedError("only flat_selection='center' is supported")
    return _islocal(A, find_min=False)


# --------------------------------------------------------------------------- #
# smooth: moving-average (default span 5)
# --------------------------------------------------------------------------- #
def smooth(y, span: int = 5):
    """Symmetric moving-average smoothing with a shrinking window at the ends.

    At each position ``i``, computes the mean over a window of half-width
    ``w = min(i, n-1-i, span//2)``, so the endpoints of ``y`` remain unchanged
    and the window narrows near the boundaries rather than padding with zeros.

    Parameters
    ----------
    y : array-like, 1-D
        Input signal to smooth.
    span : int, optional
        Full window span (number of samples); default is 5.

    Returns
    -------
    out : ndarray, shape (N,)
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    half = span // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        w = min(i, n - 1 - i, half)
        out[i] = y[i - w:i + w + 1].mean()
    return out

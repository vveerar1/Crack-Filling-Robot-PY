"""Polygon boundary refinement for the crack-filling coverage planner.

Provides :func:`refinePoly`, which densifies a polygon boundary by inserting
edge midpoints over one or more passes and optionally smoothing the result.
The densified boundary gives the Morse Cell Decomposition (MCD) enough vertices
to detect critical points (local extrema of the x-coordinate) reliably, even on
long straight edges that would otherwise be undersampled.

Two modes are supported:

- **Densification only** (``smoothflag=False``): each pass inserts a midpoint
  on every edge, doubling the vertex count.
- **Densification + smoothing** (``smoothflag=True``): after midpoint insertion,
  each region is round-cornered via an inward-then-outward polygon offset and a
  double moving-average smooth.  Holes are refined with a smaller offset radius
  and subtracted from the outer boundary.
"""

import numpy as np

from private.utils import smooth
from private import poly_utils as pu
from private.poly_utils import PolyShape


def _drop_closing(B):
    """Return an open ring by dropping the closing vertex from a closed boundary array."""
    B = np.asarray(B, dtype=float)
    if B.shape[0] >= 2 and np.allclose(B[0], B[-1]):
        B = B[:-1]
    return B


def _insert_midpoints(values):
    """Insert a midpoint on every edge of a polygon ring, doubling the vertex count.

    The output order is ``[v1, mid(v1,v2), v2, mid(v2,v3), ..., vN, mid(vN,v1)]``.
    Midpoints are computed from vertices rounded to 5 decimal places; the
    original vertices are kept unrounded.

    Parameters
    ----------
    values : array-like, shape (N, 2)
        Open ring vertices.

    Returns
    -------
    numpy.ndarray, shape (2N, 2)
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    out = np.empty((2 * n, 2), dtype=float)
    for j in range(n):
        p1 = np.round(values[j], 5)
        p2 = np.round(values[(j + 1) % n], 5)
        out[2 * j] = values[j]
        out[2 * j + 1] = (p1 + p2) / 2.0
    return out


def _join_loops(loops):
    """Concatenate a list of rings into one ``NaN``-separated array suitable for :func:`polyshape`."""
    if not loops:
        return np.empty((0, 2))
    chunks = []
    for i, l in enumerate(loops):
        if i > 0:
            chunks.append(np.array([[np.nan, np.nan]]))
        chunks.append(np.asarray(l, dtype=float))
    return np.vstack(chunks)


def _smooth_region(reg, r):
    """Smooth a single polygon region by round-cornering and averaging its boundary.

    Applies an inward offset of radius ``r``, an outward offset of the same
    radius, cleans up boundary slivers, then applies a double moving-average
    smooth to the x and y vertex coordinates.  Falls back to the unsmoothed
    region if the smoothing operation splits it into multiple parts.
    """
    g = pu.polybuffer(pu.polybuffer(reg, -r), r)
    g = pu.rmslivers(g, 1e-5)
    V = g.Vertices
    V = V[~np.isnan(V).any(axis=1)]          # guard: take finite verts
    if V.shape[0] < 3:
        return reg
    x, y = V[:, 0], V[:, 1]
    sx = smooth(smooth(x))
    sy = smooth(smooth(y))
    g2 = pu.polyshape(np.column_stack([sx, sy]))
    if len(pu.regions(g2)) > 1 or g2.is_empty:
        g2 = pu.polyshape(np.column_stack([x, y]))
    return g2


def _refine_once(ps, smoothflag):
    # outer boundary (holes removed), per region
    boundary = pu.rmholes(ps)
    ext_loops = []
    for reg in pu.regions(boundary):
        if reg.is_empty:
            continue
        g = _smooth_region(reg, 2) if smoothflag else reg
        ext_loops.append(_insert_midpoints(_drop_closing(g.boundary())))
    ext = pu.polyshape(_join_loops(ext_loops))

    # holes
    hole_loops = []
    for hreg in pu.holes(ps):
        g = _smooth_region(hreg, 1) if smoothflag else hreg
        hole_loops.append(_insert_midpoints(_drop_closing(g.boundary())))
    hol = pu.polyshape(_join_loops(hole_loops)) if hole_loops else PolyShape(None)

    return pu.subtract(ext, hol, keep_collinear=True)


def refinePoly(polyin, times, smoothflag=False):
    """Refine a polygon boundary by inserting edge midpoints, optionally with smoothing.

    Each pass inserts a midpoint on every edge, doubling the vertex count.
    Outer boundaries and holes are refined independently.

    Parameters
    ----------
    polyin : PolyShape
        Input polygon (may contain holes).
    times : int
        Number of refinement passes.  Each pass doubles the vertex count.
    smoothflag : bool, optional
        If ``True``, apply round-corner smoothing (inward-then-outward polygon
        offset + double moving-average) after each midpoint-insertion pass.
        Default is ``False``.

    Returns
    -------
    PolyShape
        Refined polygon with the same topology as ``polyin``.
    """
    cur = polyin
    for _ in range(int(times)):
        cur = _refine_once(cur, smoothflag)
    return cur

"""Reeb graph construction over Morse Cell Decomposition (MCD) cells.

The Reeb graph is the cell-adjacency structure used to plan the order in which
the robot visits and sweeps each coverage cell.  Each node in the Reeb graph is
a critical point of the MCD boundary; each edge represents one coverage cell,
connecting the two critical points that bound it (its left and right x-extrema).

Algorithm
---------
For every cell returned by :func:`MCD`:

1. Expand the cell slightly and merge it with any split edges so that critical
   points sitting exactly on a cell boundary are classified as interior.
2. Find which critical points lie inside the expanded cell.  If fewer than two
   are found, progressively grow a buffer (5 px at a time, up to 45 px) and
   retry.
3. If exactly two critical points are found, they become the endpoints of one
   Reeb edge.  If more than two are found, select the pair that is either
   (a) anchored to an already-visited critical point and maximally separated, or
   (b) simply the most widely separated pair.
4. Record the edge, the originating cell index, and three control points
   (entry critical point, cell centre, exit critical point) that describe the
   edge's mid-line curve for path-planning and visualisation.
5. Cells from which fewer than two critical points can be identified even after
   buffering are degenerate slivers and are recorded in *remreg* for the driver
   to skip.

The resulting Reeb graph is passed to :func:`reeb_traversal` (in
:mod:`ReebPath`) to compute the cell visit order.
"""

from itertools import combinations

import numpy as np

from private import poly_utils as pu
from private.utils import spdist, spdist2

_A_DEFAULT = 44   # footprint radius (px)


def _flip(P):
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return P[:, ::-1]


def Reeb(polyinreg, critP, splitEdge, a=_A_DEFAULT):
    """Build the Reeb graph over a set of MCD coverage cells.

    Parameters
    ----------
    polyinreg : PolyShape or list of PolyShape
        Coverage cells produced by :func:`MCD` (the *polyout* or *polyout_work*
        output).  Each cell should correspond to one Reeb edge.
    critP : array-like, shape (K, 2)
        Critical points in ``(y, x)`` order as returned by :func:`MCD`.  The
        function internally converts them to ``(x, y)`` for polygon interior tests.
    splitEdge : list of PolyShape or None
        Thin split-edge bars from :func:`MCD`.  Each cell is temporarily unioned
        with these bars before the interior test so that critical points on shared
        boundaries are captured by at least one cell.
    a : float, optional
        Robot footprint radius in pixels, used to intersect the cell midline and
        locate the edge centre control point.  Defaults to 44 px.

    Returns
    -------
    reebEdge : ndarray of int, shape (E, 2)
        Each row ``[i, j]`` is one Reeb edge connecting critical points *i* and
        *j* (0-based indices into *critP*).
    reebCell : ndarray of int, shape (E,)
        Index of the coverage cell (into *polyinreg*) associated with each edge.
    reeb : list of ndarray, length E
        Per-edge spline control points ``[critP[i], centre, critP[j]]``, shape
        ``(3, 2)``.  Used by the visualiser to draw the Reeb graph mid-line.
    reebwall : list of ndarray, length E
        Same as *reeb* but with the centre point shifted 50 px to the left, for
        the wall-following variant of the mid-line.
    remreg : list of int
        Indices (into *polyinreg*) of degenerate cells that could not be matched
        to two critical points and were excluded from the graph.
    """
    cells = pu.regions(polyinreg) if isinstance(polyinreg, pu.PolyShape) else list(polyinreg)
    critP = np.atleast_2d(np.asarray(critP, dtype=float))
    critP_xy = _flip(critP)                      # (x,y) for interior tests
    splitEdge = list(splitEdge) if splitEdge is not None else []

    reebEdge = []
    reebCell = []
    reeb = []
    reebwall = []
    remreg = []

    for i, cell in enumerate(cells):
        tt = cell
        if splitEdge:
            tt = pu.polybuffer(tt, 1)
            tt = pu.union([tt] + splitEdge)
            tt = pu.regCombine(tt, 1)
            regs = pu.regions(tt)
            tt = max(regs, key=lambda r: r.area)

        temp = pu.isinterior(tt, critP_xy)
        buf = 5
        while temp.sum() < 2:
            temp = pu.isinterior(pu.polybuffer(tt, buf), critP_xy)
            buf += 5
            if buf == 50:
                break
        # A cell that still has fewer than 2 interior critical points after buffering
        # is a degenerate sliver: it cannot form a valid Reeb edge (needs an entry
        # and an exit critical point), so it is recorded in remreg and skipped.
        if temp.sum() < 2:
            remreg.append(i)
            continue

        if temp.sum() > 2:
            t = np.flatnonzero(temp)
            uniq = np.unique(np.asarray(reebEdge, dtype=int)) if reebEdge else np.array([], dtype=int)
            if np.any(np.isin(t, uniq)):
                j = int(t[np.isin(t, uniq)][0])
                rest = t[t != j]
                ddd = int(np.argmax(spdist(critP[j], critP[rest])))
                keep = {j, int(rest[ddd])}
                for k in t:
                    if int(k) not in keep:
                        temp[k] = False
            else:
                comb = np.array(list(combinations(t, 2)))
                dd = spdist2(critP[comb[:, 0]], critP[comb[:, 1]])
                best = comb[np.argmax(dd)]
                for k in t:
                    if k not in best:
                        temp[k] = False

        ind = np.flatnonzero(temp)
        # cell-centre control point for the Reeb spline (visualisation)
        V = cell.Vertices
        Vx = V[~np.isnan(V[:, 0]), 0]
        Vy = V[~np.isnan(V[:, 1]), 1]
        yyy = (Vx.min() + Vx.max()) / 2.0
        try:
            in_p = pu.intersect(cell, pu.polybuffer([[yyy, Vy.min()], [yyy, Vy.max()]], a, kind="lines"))
            ipV = in_p.Vertices
            ipy = ipV[~np.isnan(ipV[:, 1]), 1]
            xxx = (ipy.min() + ipy.max()) / 2.0
        except Exception:
            xxx = (Vy.min() + Vy.max()) / 2.0

        reebEdge.append([int(ind[0]), int(ind[1])])
        reebCell.append(i)
        reeb.append(np.array([critP[ind[0]], [xxx, yyy], critP[ind[1]]]))
        reebwall.append(np.array([critP[ind[0]], [xxx, yyy - 50], critP[ind[1]]]))

    reebEdge = np.array(reebEdge, dtype=int) if reebEdge else np.empty((0, 2), dtype=int)
    reebCell = np.array(reebCell, dtype=int)
    return reebEdge, reebCell, reeb, reebwall, remreg

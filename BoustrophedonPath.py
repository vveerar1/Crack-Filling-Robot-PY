"""Per-cell boustrophedon (zig-zag) coverage path generator.

:func:`BoustrophedonPath` produces the back-and-forth sweep path inside a single
decomposed coverage cell.  Vertical sweep lines are spaced ``2 * bp_gap`` apart
across the cell's x-extent; each line is sampled between the cell's y-extent
(inset by ``bp_gap``) with alternating up/down direction, guaranteeing that every
point in the cell is within ``bp_gap`` of a waypoint.

For dead-end cells that require wall-following along their boundary before the
main sweep, an optional wall-follow prefix is prepended to the path.

This module is called by :mod:`Boustrophedon` and :mod:`Boustrophedon_CellCon`.
"""

import math

import numpy as np
from scipy.spatial import ConvexHull

from private import poly_utils as pu
from private.utils import mfix, spdist, total_length, vertical


def _colon(start, step, stop):
    """Inclusive range ``start:step:stop`` with floating-point tolerance."""
    if step == 0:
        return np.array([start])
    n = int(np.floor((stop - start) / step + 1e-9))
    if n < 0:
        return np.array([])
    return start + step * np.arange(n + 1)


def _cell_xy(cell):
    V = cell.Vertices
    V = V[~np.isnan(V).any(axis=1)]
    return V[:, 0], V[:, 1]


def BoustrophedonPath(cell, orgcell, reebEdge, bp_gap, dir, init, wall_fol,
                      known, fl_see, allNode, s, a):
    """Generate the boustrophedon coverage path for a single cell.

    Produces a back-and-forth (zig-zag) sweep inside ``cell`` at sweep spacing
    ``bp_gap``.  Sweep columns are laid out left-to-right or right-to-left
    depending on which Reeb-edge endpoint has the larger x-coordinate; the
    up/down direction of each column alternates to form a continuous path.

    When ``wall_fol`` is set the function first traces the eroded cell boundary
    (wall-follow) from the corner nearest ``init``, prepending those waypoints
    before the main sweep.

    Parameters
    ----------
    cell : PolyShape
        The coverage cell polygon to sweep (may differ from ``orgcell`` if
        already partially covered).
    orgcell : PolyShape
        The original (un-eroded) coverage cell polygon.  Used to decide whether
        to erode the boundary for wall-following.
    reebEdge : array-like, shape (2,)
        Pair of critical-point node indices ``[start, end]`` defining the Reeb
        edge associated with this cell.
    bp_gap : float
        Half the sweep-column spacing (pixels).  Columns are placed every
        ``2 * bp_gap``; the sensor guarantees coverage when ``bp_gap = s / sqrt(2)``.
    dir : int or None
        Sweep direction flag (0 = bottom-to-top first column, 1 = top-to-bottom).
        Pass ``None`` or ``[]`` to infer the direction from ``init``.
    init : array-like, shape (2,) or scalar
        Robot position (x, y) at the start of this cell.  Used to pick the
        nearer column endpoint when ``dir`` is not specified.
    wall_fol : int
        1 to prepend a wall-follow prefix; 0 for a plain sweep.
    known : bool
        ``True`` for the offline (SCC) planner; ``False`` for the online
        (OnlineSCC) planner.
    fl_see : bool
        Reserved flag (currently unused); pass ``False``.
    allNode : array-like, shape (M, 2)
        Critical-point coordinates in (row, col) order as produced by MCD.
    s : float
        Sensor radius in pixels.
    a : float
        Robot footprint radius in pixels.  Cells smaller than the footprint area
        are collapsed to their centroid.

    Returns
    -------
    subXY : ndarray, shape (K, 2)
        Ordered (x, y) sweep waypoints for this cell.
    flag : bool
        ``True`` when a standard zig-zag sweep was produced; ``False`` when
        wall-following was used and the sweep was deferred.
    """
    allNode = np.atleast_2d(np.asarray(allNode, dtype=float))
    subXY = np.empty((0, 2))
    flag = not bool(wall_fol)

    if wall_fol:
        start_pt = allNode[int(reebEdge[0])][::-1]          # (row,col) -> (x,y)
        wf = _wall_follow(cell, orgcell, bp_gap, start_pt, s, a)
        if wf.size:
            subXY = wf
        if subXY.size:
            xs = allNode[[int(reebEdge[0]), int(reebEdge[1])], 1]   # x of the two reeb nodes
            lo = int(np.argmin(np.abs(subXY[0, 0] - xs)))           # nearer of the two edge nodes
            if lo == 0:
                obj = pu.polybuffer(np.vstack([np.atleast_2d(init), subXY]),
                                    bp_gap * math.sqrt(2), kind="lines")
                cell = pu.subtract(cell, obj)
                aa = pu.regions(cell)
                if len(aa) > 1:
                    cell = max(aa, key=lambda r: r.area)
                init = subXY[-1]
                dir = None
                Start, End = int(reebEdge[1]), int(reebEdge[0])
            else:
                subXY = np.empty((0, 2))
                flag = bool(wall_fol)
                Start, End = int(reebEdge[1]), int(reebEdge[0])
        else:
            Start, End = int(reebEdge[0]), int(reebEdge[1])
    else:
        Start, End = int(reebEdge[0]), int(reebEdge[1])

    foot_area = math.pi * a * a
    if cell.area != 0 and cell.area > foot_area:
        x, y = _cell_xy(cell)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        cx, cy = cell.geom.centroid.x, cell.geom.centroid.y

        if allNode[Start, 1] > allNode[End, 1]:        # sweep right -> left
            if xmax - xmin <= 2 * bp_gap:
                jx = np.array([cx])
            else:
                jx = _colon(xmax - bp_gap, -2 * bp_gap, xmin + bp_gap)
                if jx.size == 0:
                    jx = np.array([(xmax + xmin) / 2.0])
                elif (jx[-1] - xmin) > bp_gap * math.sqrt(2):
                    jx = np.append(jx, xmin + bp_gap)
        else:                                          # sweep left -> right
            if xmax - xmin <= 2 * bp_gap:
                jx = np.array([cx])
            else:
                jx = _colon(xmin + bp_gap, 2 * bp_gap, xmax - bp_gap)
                if jx.size == 0:
                    jx = np.array([(xmax + xmin) / 2.0])
                elif (xmax - jx[-1]) > bp_gap * math.sqrt(2):
                    jx = np.append(jx, xmax - bp_gap)

        # y-extent of the cell at each sweep x (via a width-2s vertical strip)
        ins = np.zeros((2, jx.size))
        dis = np.zeros(jx.size)
        for k, jj in enumerate(jx):
            in_p = pu.intersect(cell, pu.polybuffer([[jj, ymin], [jj, ymax]], s, kind="lines"))
            iy = in_p.Vertices[:, 1]
            iy = iy[~np.isnan(iy)]
            lo, hi = iy.min(), iy.max()
            ins[:, k] = [lo, hi]
            dis[k] = total_length(np.array([[jj, lo], [jj, hi]]))

        wide = dis >= 2 * bp_gap
        ins[:, wide] += np.array([[bp_gap], [-bp_gap]])
        if np.any(~wide):
            m = ins[:, ~wide].mean(axis=0)
            ins[0, ~wide] = m
            ins[1, ~wide] = m

        if dir is None or (np.ndim(dir) == 0 and dir == []) or (hasattr(dir, "__len__") and len(dir) == 0):
            if np.ndim(init) == 0 and init == 0:
                dir_flag = False
            else:
                d = spdist(np.atleast_2d(init)[0], np.array([[jx[0], ins[0, 0]], [jx[0], ins[1, 0]]]))
                dir_flag = int(np.argmin(d)) == 1
        else:
            dir_flag = bool(dir)

        cols = np.arange(1, jx.size + 1)
        if dir_flag:
            flip = (cols % 2) != 0       # flip odd-position columns (1-based)
        else:
            flip = (cols % 2) == 0       # flip even-position columns
        ins[:, flip] = ins[::-1, flip]

        rows = [subXY] if subXY.size else []     # preserve any wall-follow prefix
        for k in range(jx.size):
            lo, hi = ins[0, k], ins[1, k]
            if lo < hi and dis[k] > bp_gap:
                sp = _colon(lo, bp_gap, hi)
                if sp.size == 0 or sp[-1] != hi:
                    sp = np.append(sp, hi)
            elif lo > hi and dis[k] > bp_gap:
                sp = _colon(lo, -bp_gap, hi)
                if sp.size == 0 or sp[-1] != hi:
                    sp = np.append(sp, hi)
            else:
                sp = ins[:, k]
            rows.append(np.column_stack([np.full(sp.size, jx[k]), sp]))
        subXY = np.vstack(rows) if rows else np.empty((0, 2))
    else:
        subXY = np.array([[cell.geom.centroid.x, cell.geom.centroid.y]])

    return subXY, flag


def _addPtsLin_we(x, y, marker_dist):
    """Densify a polyline segment by inserting waypoints at ``marker_dist`` intervals.

    Inserts one or more intermediate points along the segment defined by ``x``
    and ``y`` at arc-length multiples of ``marker_dist``.  Differs from the
    utility version in ``private.utils`` in that it inserts the first marker
    even when only one fits (the utility requires at least two).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    dist_from_start = np.concatenate(([0.0], np.cumsum(seg)))
    total = dist_from_start[-1]
    if total < marker_dist:
        return np.array([]), np.array([])
    n_locs = int(np.floor(total / marker_dist + 1e-12))
    marker_locs = marker_dist * np.arange(1, n_locs + 1)
    xv = x[~np.isnan(x)]
    if marker_locs.size == 0 or xv.size <= 1:
        return np.array([]), np.array([])
    n = dist_from_start.size
    idx_1 = np.interp(marker_locs, dist_from_start, np.arange(1, n + 1))
    base = np.floor(idx_1).astype(int)
    w = idx_1 - base
    mx = x[base - 1] * (1 - w) + x[base] * w
    my = y[base - 1] * (1 - w) + y[base] * w
    return mx, my


def _polycorner(polyin):
    """Return the convex-hull corner vertices of a polygon.

    Parameters
    ----------
    polyin : PolyShape
        Input polygon.

    Returns
    -------
    corPtx : ndarray, shape (C+1, 2)
        Corner vertices (closed — first row repeated at end).
    idx : ndarray, shape (C+1,)
        Indices into ``polyin.Vertices`` for each corner, closed in the same way.
    """
    V = polyin.Vertices
    keep = ~np.isnan(V).any(axis=1)
    Vc = V[keep]
    hull = ConvexHull(Vc)
    order = list(hull.vertices)            # CCW indices into Vc (no NaN here)
    order = order + [order[0]]             # close the hull
    base_idx = np.flatnonzero(keep)
    idx = base_idx[order]
    return V[idx], idx


def _wall_follow(polyin, orgpolyin, sensor, start, s, a):
    """Trace the eroded cell boundary for wall-following dead-end cells.

    Erodes ``polyin`` inward by ``sensor`` pixels, then walks the resulting
    boundary from the convex-hull corner nearest ``start`` until the leftmost or
    rightmost corner is reached.  The resulting sub-path is densified at spacing
    ``s`` and returned as (x, y) waypoints.

    Parameters
    ----------
    polyin : PolyShape
        The coverage cell (possibly already trimmed).
    orgpolyin : PolyShape
        The original cell before trimming.  When equal to ``polyin`` the function
        erodes the boundary itself; otherwise ``polyin`` is used directly.
    sensor : float
        Inward erosion distance (pixels); typically ``s / sqrt(2)``.
    start : array-like, shape (2,)
        Starting position (x, y) used to select the nearest corner.
    s : float
        Sensor radius in pixels; used for path densification spacing.
    a : float
        Robot footprint radius in pixels; segments shorter than ``a`` are discarded.

    Returns
    -------
    subXY : ndarray, shape (K, 2)
        Ordered (x, y) wall-follow waypoints, or an empty array if no valid path
        was found.
    """
    if np.array_equal(np.asarray(orgpolyin.Vertices), np.asarray(polyin.Vertices)):
        working = pu.polybuffer(polyin, -sensor)
        ss = sensor
        if working.area != 0:
            while True:
                if pu.area(working) == 0 or len(pu.regions(working)) > 1:
                    ss = ss - 10
                    working = pu.polybuffer(polyin, -ss)
                else:
                    break
    else:
        working = polyin

    subXY = np.empty((0, 2))
    if working.area != 0:
        corPtx, idx = _polycorner(working)
        V = working.Vertices
        N = V.shape[0]
        vertexid = int(idx[int(np.argmin(spdist(start, corPtx)))])
        temp_mask = np.abs(V[:, 0] - V[vertexid, 0]) < 2
        temp = V[temp_mask]
        t = [int(np.argmin(temp[:, 1])), int(np.argmax(temp[:, 1]))]
        t3 = int(np.argmin(spdist(V[vertexid], temp[t])))
        vertexid = int(idx[int(np.argmin(spdist(temp[t[t3]], corPtx)))])

        def _walk(step):
            sub = [V[vertexid]]
            ind = [vertexid, (vertexid + step) % N]
            if vertical(V[ind]):
                return None
            cur = ind
            while True:
                vid = cur[1]
                sub.append(V[vid])
                nxt = (cur[1] + step) % N
                cur = [cur[1], nxt]
                v0 = cur[0]
                if (np.any(v0 == idx)
                        and (mfix(V[v0, 0]) == mfix(V[:, 0].min())
                             or mfix(V[:, 0].max()) == mfix(V[v0, 0]))):
                    out = np.array(sub)
                    if out.shape[0] == 2:
                        if mfix(out[0, 0]) == mfix(out[1, 0]):
                            return np.empty((0, 2))
                        if spdist(out[0], [out[1]])[0] < a:
                            return np.empty((0, 2))
                    return out
        # try forward (next vertex), fall back to backward (prev vertex)
        res = _walk(+1)
        if res is None:
            res = _walk(-1)
        if res is not None:
            subXY = res

    if subXY.shape[0] > 1:
        if mfix(subXY[-1, 0]) == mfix(subXY[-2, 0]):
            subXY = subXY[:-1]

    # densify at spacing s
    mx2 = []
    my2 = []
    for h in range(subXY.shape[0] - 1):
        m, n = _addPtsLin_we(subXY[[h, h + 1], 0], subXY[[h, h + 1], 1], s)
        mx2.extend([subXY[h, 0]] + list(m))
        my2.extend([subXY[h, 1]] + list(n))
    if subXY.shape[0]:
        subXY = np.vstack([np.column_stack([mx2, my2]) if mx2 else np.empty((0, 2)),
                           subXY[-1]])
    return subXY

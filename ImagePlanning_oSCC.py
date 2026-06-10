"""Online crack-fill planner for the OnlineSCC coverage algorithm.

As the robot executes its boustrophedon (zig-zag) scan, its 360-degree sensor
accumulates partial crack observations.  Whenever a fillable crack is detected,
this module is called to plan the fill path for that crack and return fill
waypoints the robot can immediately execute before resuming the scan.

Given the binary crack skeleton sensed so far (``BW3``), the robot's current
position (``cp``), and the set of identified crack endpoints, this module:

1. Traces the crack skeleton into ordered polylines.
2. Filters out short stubs and polylines with no reachable real endpoint.
3. Connects nearby crack segments into chains, respecting join-angle constraints.
4. Expands each chain by the nozzle footprint radius to form buffer polygons
   (Minkowski sum).  Where polygons overlap, intersection nodes are added.
5. Builds a visibility graph over the nodes, inserting detour waypoints where
   a direct link is obstructed.
6. Routes the graph with a Chinese-Postman tour starting near the robot's
   current position, producing an ordered sequence of fill waypoints.

The ``ppath`` and ``preCrack`` parameters are accepted to keep the call
signature consistent with the full planner interface but do not affect the
output of the current implementation.

Main entry point: :func:`ImagePlanning_oSCC`.
"""

from math import atan2, cos, degrees, radians, sin, sqrt

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform

from private import poly_utils as pu
from private.utils import addPtsLin, spdist, total_length
from private.compCrack import compCrack
from private.DecimatePoly import DecimatePoly
from private.line_of_sight import line_of_sight_poly
from private.pathfinder import pathfinder
from private.Polygons_Intersection import Polygons_Intersection
from ChinesePostman import ChinesePostman

_DIR_MAP = np.array([[-1, -1], [-1, 0], [-1, 1],
                     [0, -1], [0, 1],
                     [1, -1], [1, 0], [1, 1]])


def ab2v(a, b):
    """Compute the angle from vector ``a`` to vector ``b``.

    Parameters
    ----------
    a, b : array-like, shape (2,)
        2-D vectors.

    Returns
    -------
    ang : float
        Angle in degrees, in the range [0, 360).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    theta = degrees(atan2(abs(a[1]), a[0]))
    if a[1] < 0:
        theta = 360 - theta
    tr = -radians(theta)
    rot = np.array([[cos(tr), -sin(tr)], [sin(tr), cos(tr)]])
    aR = rot @ a
    bR = rot @ b
    ang = degrees(atan2(abs(aR[0] * bR[1] - aR[1] * bR[0]), aR[0] * bR[0] + aR[1] * bR[1]))
    if bR[1] < 0:
        ang = 360 - ang
    return ang


def _dsearchn(refs, queries):
    """Find the nearest point in ``refs`` for each point in ``queries``.

    Parameters
    ----------
    refs : array-like, shape (M, 2)
        Reference point set.
    queries : array-like, shape (Q, 2)
        Query points.

    Returns
    -------
    idx : ndarray of int, shape (Q,)
        0-based index into ``refs`` of the nearest reference point.
    dist : ndarray of float, shape (Q,)
        Euclidean distance to that nearest point.
    """
    refs = np.atleast_2d(np.asarray(refs, dtype=float))
    queries = np.atleast_2d(np.asarray(queries, dtype=float))
    if refs.size == 0:
        n = queries.shape[0]
        return np.zeros(n, dtype=int), np.full(n, np.inf)
    idx = np.empty(queries.shape[0], dtype=int)
    dist = np.empty(queries.shape[0])
    for i, q in enumerate(queries):
        d = spdist(q, refs)
        k = int(np.argmin(d))
        idx[i] = k
        dist[i] = d[k]
    return idx, dist


def _ismember_rows(A, B):
    """Test whether each row of ``A`` appears in ``B``.

    Parameters
    ----------
    A : array-like, shape (M, D)
        Query rows.
    B : array-like, shape (N, D)
        Reference rows.

    Returns
    -------
    out : ndarray of bool, shape (M,)
        ``out[i]`` is True if row ``A[i]`` appears (approximately) in ``B``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.size == 0:
        return np.zeros(0, dtype=bool)
    if B.size == 0:
        return np.zeros(A.shape[0], dtype=bool)
    out = np.zeros(A.shape[0], dtype=bool)
    for i, r in enumerate(A):
        out[i] = np.any(np.all(np.isclose(B, r), axis=1))
    return out


def _filter_lists(keep, line, crackRaw, link, pointX, pointY, fP, endlogi):
    """Filter all parallel crack-data structures with a single boolean mask.

    Parameters
    ----------
    keep : array-like of bool, shape (K,)
        True to retain, False to discard.
    line, crackRaw, link, pointX, pointY, fP, endlogi
        Parallel structures of length K (see :func:`ImagePlanning_oSCC`).

    Returns
    -------
    Filtered versions of all inputs, in the same order.
    """
    keep = np.asarray(keep, dtype=bool)
    line = line[keep]
    crackRaw = [c for c, k in zip(crackRaw, keep) if k]
    link = [l for l, k in zip(link, keep) if k]
    pointX = pointX[keep] if pointX.size else pointX
    pointY = pointY[keep] if pointY.size else pointY
    fP = fP[keep] if fP.size else fP
    endlogi = endlogi[keep] if endlogi.size else endlogi
    return line, crackRaw, link, pointX, pointY, fP, endlogi


def _acute_fP(crackRaw, fP, a, yy, mark_det):
    """Adjust the footprint buffer width for sharply angled cracks.

    For each crack, the net-displacement vector is aligned with the +y axis and
    the crack's peak lateral excursion is found.  If the half opening-angle is
    smaller than 45 degrees, the effective buffer width ``fP`` is reduced from
    ``a`` to ``tan(half_angle) * a``.

    Parameters
    ----------
    crackRaw : list of ndarray
        Ordered crack polylines, each ``(M, 2)`` in pixel coordinates.
    fP : ndarray, shape (K,)
        Per-crack footprint buffer widths to update in place.
    a : int
        Nozzle footprint radius in pixels.
    yy : int
        Image width in pixels (sets the reference direction).
    mark_det : bool
        When True, also collect and return the indices of cracks that
        triggered the acute-angle condition (first-pass variant, no ``ang!=0``
        guard).  When False, only ``fP`` is updated and a zero angle is
        skipped (second-pass variant).

    Returns
    -------
    fP : ndarray
        Updated per-crack buffer widths.
    det : list of int
        Indices of acute cracks (empty when ``mark_det`` is False).
    """
    det = []
    for l, cr in enumerate(crackRaw):
        cr = np.asarray(cr, dtype=float)
        if total_length(cr) > 2 * a * sqrt(2):
            d0 = spdist(cr[0], cr)
            dN = spdist(cr[-1], cr)
            inside = (d0 <= a) | (dN <= a)
            crackW = cr[~inside]
            if crackW.shape[0] and total_length(crackW) > a:
                mov = crackW - crackW[0]
            else:
                mov = cr - cr[0]
        else:
            mov = cr - cr[0]
        v1 = np.array([0.0, yy])
        v2 = mov[-1]
        ang = ab2v(v1, v2)
        ar = radians(ang)
        rot = np.array([[cos(ar), sin(ar)], [-sin(ar), cos(ar)]])
        rv2 = (rot @ mov.T).T
        i_min = int(np.argmin(rv2[:, 0]))
        i_max = int(np.argmax(rv2[:, 0]))
        cand = [i_min, i_max]
        peak = cand[int(np.argmax(np.abs(rv2[cand, 0])))]
        v1 = -rv2[peak]
        v2 = rv2[-1] - rv2[peak]
        ang = ab2v(v1, v2) / 2
        if mark_det:
            if ang < 45 or ang > 360 - 45:
                fP[l] = np.tan(radians(ang)) * a
                det.append(l)
        else:
            if (ang < 45 or ang > 360 - 45) and ang != 0:
                fP[l] = np.tan(radians(ang)) * a
    return fP, det


def _clip_outside(ps, pts):
    """Return the portion of a polyline that lies outside a polygon.

    Clips the polyline ``pts`` against the polygon ``ps`` and returns the
    pieces that fall outside, concatenated in order along the polyline.

    Parameters
    ----------
    ps : PolyShape
        Clipping polygon (nozzle footprint buffer).
    pts : array-like, shape (M, 2)
        Ordered polyline vertices.

    Returns
    -------
    out : ndarray, shape (R, 2)
        Vertices of the outside portion, ordered along the original polyline.
        Returns an empty array if the entire polyline falls inside the polygon.
    """
    from shapely.geometry import LineString, Point
    g = pu._as_geom(ps)
    line = LineString(np.atleast_2d(np.asarray(pts, dtype=float)))
    if g is None or g.is_empty:
        return np.asarray(pts, dtype=float)
    out = line.difference(g)
    if out.is_empty:
        return np.empty((0, 2))
    if out.geom_type == "LineString":
        comps = [out]
    else:
        comps = [c for c in out.geoms if c.geom_type == "LineString" and not c.is_empty]
    pieces = []
    for c in comps:
        coords = np.asarray(c.coords, dtype=float)
        pos = line.project(Point(coords[0]))
        pieces.append((pos, coords))
    pieces.sort(key=lambda s: s[0])
    if not pieces:
        return np.empty((0, 2))
    return np.vstack([p[1] for p in pieces])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def ImagePlanning_oSCC(BW3, a, s, cp, acp, endP, realEndP, contEndP,
                       ppath=None, preCrack=None):
    """Plan the fill path for a crack detected during online scanning.

    Extracts the crack structure from the partial skeleton image ``BW3``,
    connects crack segments into chains, builds a visibility graph over
    fill-waypoint nodes, routes a Chinese-Postman tour through it, and
    assembles the ordered fill waypoints for the robot's nozzle.

    Parameters
    ----------
    BW3 : ndarray or sparse matrix
        Binary image of the crack skeleton sensed so far.  Non-zero pixels
        are treated as crack.
    a : int
        Nozzle footprint radius in pixels.
    s : int
        Sensor detection radius in pixels.
    cp : array-like, shape (2,)
        Robot's current position in pixel coordinates.
    acp : array-like, shape (2,)
        Robot's position at the start of the current scan pass.
    endP : array-like, shape (E, 2) or empty
        Skeleton endpoints (pixel coordinates) identified in the current
        sensor window.
    realEndP : array-like, shape (R, 2) or empty
        Skeleton endpoints that lie on real crack tips (not continuation
        points), used to filter which crack segments to plan.
    contEndP : array-like, shape (C, 2) or empty
        Continuation endpoints (crack pixels at the edge of the sensor
        window that continue beyond the currently sensed area); used to
        exclude from the Chinese-Postman start-node selection.
    ppath : ignored
        Accepted for interface compatibility; has no effect.
    preCrack : ignored
        Accepted for interface compatibility; has no effect.

    Returns
    -------
    waypoint_coords : ndarray, shape (N, 3)
        Ordered fill waypoints: each row is ``[row, col, fill_flag]`` where
        ``fill_flag`` is 1 when the nozzle should be dispensing and 0 during
        transit moves.
    flag : int
        1 if a fillable crack was found and waypoints were produced; 0 if
        no fillable crack was found (caller should not splice the waypoints).
    crackR : list of ndarray
        The crack polylines processed during this call.
    ttt : float
        Timing placeholder (always 0.0).
    """
    ttt = 0.0
    waypoint_coords = np.empty((0, 3))
    flag = 0
    crackR = []

    BW3 = np.asarray(BW3.todense()) if hasattr(BW3, "todense") else np.asarray(BW3)
    BW3 = (BW3 > 0).astype(int)
    if BW3.sum() <= a / 4:
        return waypoint_coords, flag, [], ttt

    cp = np.asarray(cp, dtype=float).ravel()
    acp = np.asarray(acp, dtype=float).ravel()
    endP = np.atleast_2d(np.asarray(endP, dtype=int)) if np.size(endP) else np.empty((0, 2), int)
    realEndP = np.atleast_2d(np.asarray(realEndP, dtype=float)) if np.size(realEndP) else np.empty((0, 2))
    contEndP = np.atleast_2d(np.asarray(contEndP, dtype=float)) if np.size(contEndP) else np.empty((0, 2))

    BW3 = np.pad(BW3, ((0, 1), (0, 1)))   # one-pixel zero border at bottom and right
    I = BW3.copy()

    crackRaw, line, pointX, pointY = compCrack(I, endP, _DIR_MAP, None)
    line = np.atleast_2d(np.asarray(line, dtype=float)) if len(crackRaw) else np.empty((0, 4))

    # remove cracks shorter than a
    c_chq = np.array([total_length(np.asarray(c, float)) < a for c in crackRaw], dtype=bool)
    keep = ~c_chq
    crackRaw = [c for c, k in zip(crackRaw, keep) if k]
    line = line[keep]
    pointX = pointX[keep]
    pointY = pointY[keep]

    # ---- filter: keep only segments that have at least one real crack endpoint ----
    K = line.shape[0]
    if realEndP.shape[0]:
        starts = line[:, [0, 1]]
        ends = line[:, [2, 3]]
        allpts = np.vstack([starts, ends])           # 2K x 2 (row,col)
        flagvec = np.zeros(allpts.shape[0], dtype=bool)
        for j, p in enumerate(allpts):
            flagvec[j] = np.any(spdist(p, realEndP) < 5)
        endlogi = np.column_stack([flagvec[:K], flagvec[K:]])   # K x 2
    else:
        endlogi = np.zeros((K, 2), dtype=bool)

    keep = endlogi[:, 0] | endlogi[:, 1]
    line = line[keep]
    crackRaw = [c for c, k in zip(crackRaw, keep) if k]
    pointX = pointX[keep]
    pointY = pointY[keep]
    endlogi = endlogi[keep]

    link = [{"x": None, "y": None} for _ in range(line.shape[0])]

    # ---- acute-angle pass: compute per-crack buffer widths; hold acute edges with real endpoints ----
    yy = BW3.shape[1]
    fP = np.full(line.shape[0], float(a))
    fP, det = _acute_fP(crackRaw, fP, a, yy, mark_det=True)

    # retain acute cracks only when they touch at least one real endpoint
    det = [l for l in det if endlogi[l].sum() > 0]
    h_line = np.empty((0, 4))
    h_crackRaw = []
    h_pointX = np.empty((0, pointX.shape[1] if pointX.size else 24))
    h_pointY = np.empty((0, pointY.shape[1] if pointY.size else 24))
    h_fP = np.empty(0)
    if det:
        dmask = np.zeros(line.shape[0], dtype=bool)
        dmask[det] = True
        h_line = line[dmask]
        h_pointX = pointX[dmask]
        h_pointY = pointY[dmask]
        h_crackRaw = [c for c, k in zip(crackRaw, dmask) if k]
        h_fP = fP[dmask]
        keep = ~dmask
        line, crackRaw, link, pointX, pointY, fP, endlogi = _filter_lists(
            keep, line, crackRaw, link, pointX, pointY, fP, endlogi)

    # ---- connect crack graph and build fill route ----
    if line.shape[0] or h_line.shape[0]:
        if line.shape[0]:
            line, crackRaw, link, pointX, pointY, fP, endlogi = _connect_graph(
                line, crackRaw, link, pointX, pointY, fP, endlogi,
                realEndP, cp, acp, s, a)

        # re-add the held acute edges
        if h_line.shape[0]:
            line = np.vstack([line, h_line]) if line.size else h_line.copy()
            crackRaw = crackRaw + h_crackRaw
            fP = np.concatenate([fP, h_fP]) if fP.size else h_fP.copy()
            for kk in range(h_line.shape[0]):
                link.append({"x": h_pointX[kk][::-1].copy(), "y": h_pointY[kk][::-1].copy()})

        crackR = crackRaw

        # acute-angle pass 2: refine fP (no new detections)
        fP, _ = _acute_fP(crackRaw, fP, a, yy, mark_det=False)

        if line.shape[0]:
            waypoint_coords, flag = _build_and_route(
                line, crackRaw, link, fP, endlogi, realEndP, contEndP, cp, acp, a, s)
        else:
            waypoint_coords, flag = np.empty((0, 3)), 0
    else:
        waypoint_coords, flag, crackR = np.empty((0, 3)), 0, []

    return waypoint_coords, flag, crackR, ttt


def _connect_graph(line, crackRaw, link, pointX, pointY, fP, endlogi,
                   realEndP, cp, acp, s, a):
    """Connect nearby crack segments into chains, anchored to real endpoints.

    Selects which crack segment to start from based on its distance to the
    robot's current position and whether its endpoints are identified as real
    crack tips.  Attempts to merge adjacent segments by chaining them end-to-end
    when the gap is within ``a`` pixels and the join angle is not too sharp.
    Segments that cannot be connected are discarded (they will be replanned on a
    later call when the robot is closer).

    Parameters
    ----------
    line : ndarray, shape (K, 4)
        Start/end coordinates of each crack segment: ``[r0, c0, r1, c1]``.
    crackRaw : list of ndarray
        Ordered crack polylines.
    link : list of dict
        Per-segment waypoint lists (``{"x": ..., "y": ...}``); populated here.
    pointX, pointY : ndarray, shape (K, P)
        Pre-sampled waypoint coordinates for each segment.
    fP : ndarray, shape (K,)
        Per-segment footprint buffer widths.
    endlogi : ndarray of bool, shape (K, 2)
        Which end (start=0, end=1) of each segment is a real crack endpoint.
    realEndP : ndarray, shape (R, 2)
        Real crack-tip pixel coordinates.
    cp : ndarray, shape (2,)
        Robot's current position.
    acp : ndarray, shape (2,)
        Robot's position at the start of this scan pass.
    s : int
        Sensor radius in pixels.
    a : int
        Nozzle footprint radius (maximum merge gap) in pixels.

    Returns
    -------
    line, crackRaw, link, pointX, pointY, fP, endlogi
        Filtered and partially merged versions of the inputs.
    """
    detind = []
    templine = line.copy()
    cp_eq_acp = np.allclose(cp, acp)

    starts = line[:, [0, 1]]
    ends = line[:, [2, 3]]
    allpts = np.vstack([starts, ends])
    K = line.shape[0]

    endlogi_t = (spdist(cp, allpts) < s * sqrt(2)).reshape(2, K).T & endlogi
    ll = endlogi.sum(axis=1) > 0

    tt = np.vstack([templine[ll][:, [0, 1]], templine[ll][:, [2, 3]]])

    if cp_eq_acp:
        sums = endlogi_t.sum(axis=1)
        ll = sums == sums.max()

    endlogi_t2 = endlogi.copy()
    endlogi_t2[~ll] = 0
    ee_mat = endlogi_t2
    ee_flat = np.concatenate([ee_mat[:, 0], ee_mat[:, 1]]).astype(bool)
    tt2 = tt[ee_flat[:tt.shape[0]]] if tt.shape[0] else np.empty((0, 2))

    def idx_in_tt(row):
        m = np.all(np.isclose(tt, row), axis=1)
        return int(np.flatnonzero(m)[0]) if m.any() else -1

    if tt2.shape[0] == 0:
        detind = list(range(line.shape[0]))
        return _apply_detind(detind, line, crackRaw, link, pointX, pointY, fP, endlogi)

    d_cp = spdist(cp, tt2)
    ee1 = int(np.argmin(d_cp))
    cd = float(d_cp[ee1])
    ee = idx_in_tt(tt2[ee1])
    # skip intersection points (shared by more than one segment)
    while np.sum(np.all(np.isclose(tt, tt[ee]), axis=1)) > 1:
        dupmask = np.all(np.isclose(tt2, tt2[ee1]), axis=1)
        tt2 = tt2[~dupmask]
        if tt2.shape[0] == 0:
            break
        d_cp = spdist(cp, tt2)
        ee1 = int(np.argmin(d_cp))
        cd = float(d_cp[ee1])
        ee = idx_in_tt(tt2[ee1])

    if cd <= s * sqrt(2):
        # nearest realEndP to chosen endpoint
        de = spdist(tt[ee], realEndP)
        ee_re = de == de.min()
        _, dd = _dsearchn(realEndP[ee_re], np.vstack([line[:, [0, 1]], line[:, [2, 3]]]))
        dd = (dd < 6).reshape(2, K).T
        zero = ~((dd.sum(axis=1) > 0) & ll)
        dd[zero] = False
        ll2 = np.flatnonzero(dd.sum(axis=1))
        if ll2.size > 1:
            pp1 = []
            for lll in ll2:
                _, pp = _dsearchn(crackRaw[lll], acp)
                pp1.append(float(pp[0]))
            pp1 = np.array(pp1)
            kill = ll2[pp1 != pp1.min()]
            dd[kill] = False

        if dd.any():
            col0 = np.flatnonzero(dd[:, 0])
            if col0.size == 0:
                intt = int(np.flatnonzero(dd[:, 1])[0])
            else:
                intt = int(col0[0])
            crackRawt = [np.asarray(c, float).copy() for c in crackRaw]
            if int(np.flatnonzero(dd[intt])[0]) == 0:
                link[intt]["x"] = pointX[intt].copy()
                link[intt]["y"] = pointY[intt].copy()
            else:
                link[intt]["x"] = pointX[intt][::-1].copy()
                link[intt]["y"] = pointY[intt][::-1].copy()
                endlogi[intt] = endlogi[intt][::-1]
                crackRaw[intt] = np.asarray(crackRaw[intt], float)[::-1].copy()
                line[intt] = np.concatenate([line[intt, [2, 3]], line[intt, [0, 1]]])
            templine[intt] = 0

            if endlogi[intt, 1] != 0 and cp_eq_acp:
                t_starts = templine[:, [0, 1]]
                t_ends = templine[:, [2, 3]]
                el2 = np.column_stack([
                    _ismember_rows(t_starts, realEndP),
                    _ismember_rows(t_ends, realEndP)])
                ss = el2.sum(axis=1)
                if np.any(ss == 2):
                    detind = _connect_multi(intt, intt, ss, el2, templine, line, crackRaw,
                                            crackRawt, link, pointX, pointY, a, detind)
                elif np.all(ss == 0):
                    detind += list(np.flatnonzero(np.arange(line.shape[0]) != intt))
                else:
                    ddist = []
                    for lq in np.flatnonzero((np.arange(el2.shape[0]) != intt) & (ss > 0)):
                        ii = int(np.flatnonzero(el2[lq])[0])
                        ddist.append([lq, float(spdist(
                            [link[intt]["x"][-1], link[intt]["y"][-1]],
                            templine[lq, [2 * ii, 2 * ii + 1]])[0])])
                    ddist = np.array(ddist)
                    m = int(np.argmin(ddist[:, 1]))
                    n = ddist[m, 1]
                    if n <= a:
                        src = int(ddist[m, 0])
                        if int(np.flatnonzero(el2[src])[0]) == 0:
                            link[intt]["x"] = np.concatenate([link[intt]["x"], pointX[src]])
                            link[intt]["y"] = np.concatenate([link[intt]["y"], pointY[src]])
                            crackRaw[intt] = np.vstack([crackRaw[intt], crackRaw[src]])
                        else:
                            link[intt]["x"] = np.concatenate([link[intt]["x"], pointX[src][::-1]])
                            link[intt]["y"] = np.concatenate([link[intt]["y"], pointY[src][::-1]])
                            crackRaw[intt] = np.vstack([crackRaw[intt], np.asarray(crackRaw[src], float)[::-1]])
                    detind += list(np.flatnonzero(np.arange(line.shape[0]) != intt))
            elif endlogi[intt, 1] == 0 or not cp_eq_acp:
                detind += list(np.flatnonzero(np.arange(line.shape[0]) != intt))

    elif endlogi.sum(axis=1).max() == 2 and cp_eq_acp:
        detind = list(range(line.shape[0]))
    else:
        detind = list(range(line.shape[0]))

    return _apply_detind(detind, line, crackRaw, link, pointX, pointY, fP, endlogi)


def _connect_multi(intt, indt, ss, el2, templine, line, crackRaw, crackRawt,
                   link, pointX, pointY, a, detind):
    """Stitch multiple two-endpoint crack segments into a single connected chain.

    Starting from the current chain head ``intt``, repeatedly extends the chain
    by attaching the nearest unattached segment whose start or end is within
    ``a`` pixels of the chain tail and whose join angle is not too sharp (not
    acute).  Segments with no acceptable neighbour start a new independent chain
    head.  All consumed segments accumulate in ``detind`` for later removal.

    Mutates ``line``, ``crackRaw``, ``link``, and ``templine`` in place.

    Parameters
    ----------
    intt : int
        Index of the current chain head segment.
    indt : int
        Index used for direction-vector reference (initially equal to ``intt``).
    ss : ndarray of int, shape (K,)
        Number of real endpoints for each segment (0, 1, or 2).
    el2 : ndarray of bool, shape (K, 2)
        Which end (0=start, 1=end) of each segment is a real crack endpoint.
    templine : ndarray, shape (K, 4)
        Working copy of ``line``; consumed segments are zeroed out.
    line : ndarray, shape (K, 4)
        Segment start/end coordinates; updated in place.
    crackRaw : list of ndarray
        Crack polylines; extended in place.
    crackRawt : list of ndarray
        Reference copies of ``crackRaw`` for direction-vector computation.
    link : list of dict
        Per-segment waypoint chains; extended in place.
    pointX, pointY : ndarray, shape (K, P)
        Pre-sampled waypoint coordinates.
    a : int
        Maximum gap in pixels for connecting two segments.
    detind : list of int
        Accumulator for consumed segment indices.

    Returns
    -------
    detind : list of int
        Updated list of consumed segment indices.
    """
    nlines = line.shape[0]
    intt2 = list(np.flatnonzero(ss == 2))
    detind = detind + list(np.flatnonzero(
        ~np.isin(np.arange(nlines), intt2) & (np.arange(nlines) != intt)))
    keep_t = np.zeros(nlines, dtype=bool)
    keep_t[intt2] = True
    templine[~keep_t] = 0

    def _ang_ok(ang):
        return not (ang < 45 or ang > 360 - 45)

    while True:
        tail = np.array([link[intt]["x"][-1], link[intt]["y"][-1]], float)
        tempdist = spdist(tail, templine[:, [0, 1]])          # to starts
        inst_set = np.flatnonzero(tempdist <= a)
        connected = False
        if inst_set.size:
            inst = int(np.flatnonzero(tempdist == tempdist[inst_set].min())[0])
            v1 = np.asarray(crackRawt[indt], float)[0] - np.asarray(crackRawt[indt], float)[-1]
            v2 = np.asarray(crackRaw[intt], float)[0] - np.asarray(crackRaw[intt], float)[-1]
            v3 = np.asarray(crackRaw[inst], float)[-1] - np.asarray(crackRaw[inst], float)[0]
            ang = ab2v(v1, v3)
            indt = inst
            ang2 = ab2v(v2, v3)
            if _ang_ok(ang) and _ang_ok(ang2):
                d = int(np.argmin(tempdist))
                link[intt]["x"] = np.concatenate([link[intt]["x"], pointX[d]])
                link[intt]["y"] = np.concatenate([link[intt]["y"], pointY[d]])
                line[intt, 2:4] = [link[intt]["x"][-1], link[intt]["y"][-1]]
                crackRaw[intt] = np.vstack([crackRaw[intt], crackRaw[d]])
                templine[d] = 0
                detind = detind + [d]
                if d in intt2:
                    intt2.remove(d)
                connected = True
        if not connected:
            tempdist = spdist(tail, templine[:, [2, 3]])      # to ends
            inst_set = np.flatnonzero(tempdist <= a)
            attached = False
            if inst_set.size:
                inst = int(np.flatnonzero(tempdist == tempdist[inst_set].min())[0])
                v1 = np.asarray(crackRawt[indt], float)[0] - np.asarray(crackRawt[indt], float)[-1]
                v2 = np.asarray(crackRaw[intt], float)[0] - np.asarray(crackRaw[intt], float)[-1]
                v3 = np.asarray(crackRaw[inst], float)[0] - np.asarray(crackRaw[inst], float)[-1]
                ang = ab2v(v1, v3)
                indt = inst
                ang2 = ab2v(v2, v3)
                if _ang_ok(ang) and _ang_ok(ang2):
                    d = int(np.argmin(tempdist))
                    link[intt]["x"] = np.concatenate([link[intt]["x"], pointX[d][::-1]])
                    link[intt]["y"] = np.concatenate([link[intt]["y"], pointY[d][::-1]])
                    line[intt, 2:4] = [link[intt]["x"][-1], link[intt]["y"][-1]]
                    crackRaw[intt] = np.vstack([crackRaw[intt], np.asarray(crackRaw[d], float)[::-1]])
                    crackRawt[d] = np.asarray(crackRawt[d], float)[::-1]
                    templine[d] = 0
                    detind = detind + [d]
                    if d in intt2:
                        intt2.remove(d)
                    attached = True
            if not attached:
                intt = intt2[0]
                indt = intt
                link[intt]["x"] = np.concatenate([np.asarray(link[intt]["x"] if link[intt]["x"] is not None else [], float), pointX[intt]])
                link[intt]["y"] = np.concatenate([np.asarray(link[intt]["y"] if link[intt]["y"] is not None else [], float), pointY[intt]])
                line[intt, 2:4] = [link[intt]["x"][-1], link[intt]["y"][-1]]
                templine[intt] = 0
                if intt in intt2:
                    intt2.remove(intt)
        if np.any(templine):
            continue
        break
    return detind


def _apply_detind(detind, line, crackRaw, link, pointX, pointY, fP, endlogi):
    """Remove consumed/discarded segment indices from all parallel data structures.

    Parameters
    ----------
    detind : list of int
        Indices to remove.
    line, crackRaw, link, pointX, pointY, fP, endlogi
        Parallel crack-data structures of length K.

    Returns
    -------
    Filtered versions of all inputs with the listed indices removed, in the
    same order as the inputs.
    """
    detind = sorted(set(int(d) for d in detind))
    keep = np.ones(line.shape[0], dtype=bool)
    if detind:
        keep[detind] = False
    line = line[keep]
    crackRaw = [c for c, k in zip(crackRaw, keep) if k]
    pointX = pointX[keep]
    pointY = pointY[keep]
    fP = fP[keep]
    endlogi = endlogi[keep]
    link = [l for l in link if not (l["x"] is None)]
    return line, crackRaw, link, pointX, pointY, fP, endlogi


def _build_and_route(line, crackRaw, link, fP, endlogi, realEndP, contEndP, cp, acp, a, s):
    """Build the visibility graph and compute the fill-waypoint route.

    Constructs buffer polygons around each crack chain (Minkowski sum),
    identifies overlap nodes where buffers intersect, builds a visibility graph
    that connects all nodes, then routes a Chinese-Postman tour starting near
    the robot's current position.  Assembles the final waypoint sequence with
    transit and fill flags.

    Parameters
    ----------
    line : ndarray, shape (K, 4)
        Connected crack segments: ``[r0, c0, r1, c1]``.
    crackRaw : list of ndarray
        Crack polylines for the connected segments.
    link : list of dict
        Per-segment waypoint chains (``{"x": ..., "y": ...}``).
    fP : ndarray, shape (K,)
        Per-segment footprint buffer widths.
    endlogi : ndarray of bool, shape (K, 2)
        Real-endpoint flags for each segment end.
    realEndP : ndarray, shape (R, 2)
        Real crack-tip coordinates.
    contEndP : ndarray, shape (C, 2)
        Continuation endpoint coordinates (excluded from tour start selection).
    cp : ndarray, shape (2,)
        Robot's current position.
    acp : ndarray, shape (2,)
        Robot's position at the start of this scan pass.
    a : int
        Nozzle footprint radius in pixels.
    s : int
        Sensor radius in pixels.

    Returns
    -------
    waypoint_coords : ndarray, shape (N, 3)
        Ordered fill waypoints ``[row, col, fill_flag]``.
    flag : int
        1 if a valid fill route was produced; 0 otherwise.
    """
    K = line.shape[0]

    # ---- re-check real endpoint membership after merge ----
    starts = line[:, [0, 1]]
    ends = line[:, [2, 3]]
    el = np.column_stack([_ismember_rows(starts, realEndP), _ismember_rows(ends, realEndP)])
    _, dd = _dsearchn(realEndP, np.vstack([starts, ends]))
    el = el | (dd < 6).reshape(2, K).T
    keep = el[:, 0] | el[:, 1]
    line = line[keep]
    crackRaw = [c for c, k in zip(crackRaw, keep) if k]
    link = [l for l, k in zip(link, keep) if k]
    fP = fP[keep]
    K = line.shape[0]

    # ---- endpoint nodes ----
    starts = line[:, [0, 1]]
    ends = line[:, [2, 3]]
    allpts = np.vstack([starts, ends])
    endPoints = np.unique(allpts, axis=0)
    endPoints_t = endPoints.copy()
    endNodes = []
    while endPoints_t.shape[0]:
        e0 = endPoints_t[0]
        belong = _ismember_rows(allpts, e0).reshape(2, K).T.sum(axis=1) > 0
        owned = np.vstack([line[belong][:, [0, 1]], line[belong][:, [2, 3]]]) if belong.any() else np.empty((0, 2))
        keepmask = ~_ismember_rows(endPoints_t, owned) if owned.size else np.ones(endPoints_t.shape[0], bool)
        TestPoints = endPoints_t[keepmask]
        if TestPoints.shape[0] == 0 or np.all(spdist(e0, TestPoints) > a):
            endNodes.append(e0)
        endPoints_t = endPoints_t[1:]
    endNodes = np.array(endNodes) if endNodes else np.empty((0, 2))

    # ---- Minkowski sum: buffer each crack chain by its footprint width ----
    S = []        # decimated buffer polygons
    v = []        # same polys (for overlap detection)
    u = None
    for i in range(K):
        lx = np.asarray(link[i]["x"], float)
        ly = np.asarray(link[i]["y"], float)
        poly = pu.polybuffer(np.column_stack([ly, lx]), fP[i], kind="lines")  # (x=col,y=row)
        bd = poly.boundary()
        px, py = bd[:, 0], bd[:, 1]
        dec = DecimatePoly(np.column_stack([px, py]), [1, 1], False)
        if isinstance(dec, tuple):
            dec = dec[0]
        dec = np.asarray(dec, float)
        poly = pu.polyshape(dec[:, 0], dec[:, 1])
        S.append(poly)
        v.append(poly)
        u = poly.geom if u is None else u.union(poly.geom)

    # ---- overlapping buffers -> intersection nodes ----
    overlap = False
    for i in range(K):
        for j in range(K):
            if i != j and v[i].geom is not None and v[j].geom is not None:
                if v[i].geom.intersection(v[j].geom).area > 1e-9:
                    overlap = True
                    break
        if overlap:
            break

    if overlap:
        Geo = Polygons_Intersection(S, 0, 1e-3)
        Ind = []
        nodes = []
        for g in Geo:
            if len(g["index"]) >= 2:
                region = g["P"]
                parts = list(region.geoms) if region.geom_type.startswith("Multi") else [region]
                for part in parts:
                    Ind.append(len(g["index"]))
                    c = part.centroid
                    nodes.append([c.y, c.x])     # [cx,cy]=[row,col]=(row,col)
        nodes = np.array(nodes) if nodes else np.empty((0, 2))
        if nodes.shape[0]:
            order = np.argsort(-np.asarray(Ind), kind="stable")
            SortNode = nodes[order]
        else:
            SortNode = np.empty((0, 2))
        NodeCan = np.vstack([SortNode, endNodes]) if endNodes.size or SortNode.size else np.empty((0, 2))
        node = []
        for nc in NodeCan:
            if not node or np.all(spdist(nc, np.array(node)) > a):
                node.append(nc)
        node = np.array(node) if node else np.empty((0, 2))
    else:
        node = endNodes.copy()

    # ---- shorten endpoint nodes to the edge of each crack's footprint ----
    realEndP_both = np.vstack([realEndP, realEndP[:, [1, 0]]]) if realEndP.size else np.empty((0, 2))
    for e in range(node.shape[0]):
        aRan = pu.polybuffer(node[e], a / sqrt(2), kind="points")
        logi = np.array([_intersect_nonempty(aRan, c) for c in crackRaw], dtype=bool)
        if logi.sum() == 1 and _ismember_rows(node[e][None, :], realEndP_both)[0]:
            ci = int(np.flatnonzero(logi)[0])
            cr = np.asarray(crackRaw[ci], float)
            if np.any(pu.isinterior(aRan, cr[[0, -1]])):
                out = _clip_outside(aRan, cr)
                if out.shape[0]:
                    crackRaw[ci] = out
                    ee_pts = out[[0, -1]]
                    sel = spdist(node[e], ee_pts)
                    node[e] = ee_pts[int(np.argmin(sel))]

    # ---- visibility graph ----
    vgNE = [None] * len(S)
    vgEE = [None] * len(S)
    iT = []
    for i in range(len(S)):
        Vx = S[i].Vertices[:, 0]      # x = col
        Vy = S[i].Vertices[:, 1]      # y = row
        IN, _ = pu.inpolygon(node[:, 0], node[:, 1], Vy, Vx)
        if IN.any():
            insidx = np.flatnonzero(IN)
            VP = np.column_stack([insidx, node[IN]])          # [idx,row,col]
            linkpts = np.column_stack([np.asarray(link[i]["x"], float),
                                       np.asarray(link[i]["y"], float)])
            mem = _ismember_rows(VP[:, 1:3], linkpts)
            ksrch, dsr = _dsearchn(linkpts, VP[:, 1:3])
            mem = mem | (dsr < a)
            if mem.any():
                ks, _ = _dsearchn(linkpts, VP[mem, 1:3])
                tab = np.column_stack([ks, VP[mem]])
                tab = tab[np.argsort(tab[:, 0], kind="stable")]
            else:
                tab = np.empty((0, 4))
            vne = []
            vee = []
            for j in range(tab.shape[0] - 1):
                vne.append([tab[j, 2], tab[j, 3], tab[j + 1, 2], tab[j + 1, 3]])
                vee.append([tab[j, 1], tab[j + 1, 1]])
            # ~mem nodes: connect to nearest sorted node
            if (~mem).any() and tab.shape[0]:
                t2, _ = _dsearchn(tab[:, 2:4], VP[~mem, 1:3])
                for q, nidx in zip(np.flatnonzero(~mem), t2):
                    vne.append([VP[q, 1], VP[q, 2], tab[nidx, 2], tab[nidx, 3]])
                    vee.append([VP[q, 0], tab[nidx, 1]])
            vgNE[i] = np.array(vne) if vne else np.empty((0, 4))
            vgEE[i] = np.array(vee) if vee else np.empty((0, 2))
        else:
            iT.append(i)
            vgNE[i] = np.empty((0, 4))
            vgEE[i] = np.empty((0, 2))

    # remove S/vg entries with no interior nodes
    keepS = [i for i in range(len(S)) if i not in iT]
    S = [S[i] for i in keepS]
    vgNE = [vgNE[i] for i in keepS]
    vgEE = [vgEE[i] for i in keepS]

    for i in range(len(S)):
        if vgNE[i].shape[0] == 0:
            continue
        bnd = np.vstack([S[i].Vertices[:, 1], S[i].Vertices[:, 0]])  # [row; col]
        vis = line_of_sight_poly(vgNE[i][:, 0:2], vgNE[i][:, 2:4], bnd)
        for j in range(len(vis)):
            if not vis[j]:
                start = vgNE[i][j, 0:2]
                goal = vgNE[i][j, 2:4]
                boundary = np.column_stack([S[i].Vertices[:, 1], S[i].Vertices[:, 0]])
                startn = vgEE[i][j, 0]
                goaln = vgEE[i][j, 1]
                slen = node.shape[0]
                wp, _ = pathfinder(start, goal, boundary)
                nn = wp[1:-1] if wp.shape[0] > 2 else np.empty((0, 2))
                if nn.shape[0]:
                    node = np.vstack([node, nn])
                    vgNE[i] = np.vstack([vgNE[i], np.concatenate([start, nn[0]])])
                    vgEE[i] = np.vstack([vgEE[i], [startn, slen]])
                    for k in range(1, nn.shape[0]):
                        vgNE[i] = np.vstack([vgNE[i], np.concatenate([vgNE[i][-1, 2:4], nn[k]])])
                        vgEE[i] = np.vstack([vgEE[i], [vgEE[i][-1, 1], slen + k]])
                    vgNE[i] = np.vstack([vgNE[i], np.concatenate([vgNE[i][-1, 2:4], goal])])
                    vgEE[i] = np.vstack([vgEE[i], [vgEE[i][-1, 1], goaln]])
                else:
                    vgNE[i] = np.vstack([vgNE[i], np.concatenate([start, goal])])
                    vgEE[i] = np.vstack([vgEE[i], [startn, goaln]])
        keepe = vis.astype(bool)
        # appended pathfinder edges are always kept; only the original direct edges are filtered
        nkeep = keepe.shape[0]
        mask = np.ones(vgEE[i].shape[0], dtype=bool)
        mask[:nkeep] = keepe
        vgNE[i] = vgNE[i][mask]
        vgEE[i] = vgEE[i][mask]

    edgeList = np.empty((0, 2))
    for i in range(len(vgEE)):
        if vgEE[i].size:
            edgeList = np.vstack([edgeList, vgEE[i]]) if edgeList.size else np.asarray(vgEE[i], float)
    edgeList = edgeList.astype(int)

    edgeList = np.sort(edgeList, axis=1)
    edgeList = np.unique(edgeList, axis=0) if edgeList.size else edgeList

    # build graph; connect any isolated nodes to their nearest neighbour
    G = nx.Graph()
    G.add_nodes_from(range(node.shape[0]))
    for u_, v_ in edgeList:
        G.add_edge(int(u_), int(v_))
    while True:
        have = set()
        for a_, b_ in G.edges():
            have.add(a_)
            have.add(b_)
        missing = [n for n in range(node.shape[0]) if n not in have]
        if not missing:
            break
        dvals = spdist(cp, node[missing])
        nnode = missing[int(np.argmin(dvals))]
        dd2 = spdist(node[nnode], node)
        dd2[dd2 == 0] = np.inf
        tgt = int(np.argmin(dd2))
        G.add_edge(nnode, tgt)
        edgeList = np.vstack([edgeList, [nnode, tgt]]) if edgeList.size else np.array([[nnode, tgt]])

    # ---- Chinese-Postman route ----
    if edgeList.shape[0] > 1:
        remNode = np.flatnonzero(_ismember_rows(node, contEndP)) if contEndP.size else np.empty(0, int)
        eeP = np.vstack([line[:, [0, 1]], line[:, [2, 3]]])
        dline = spdist(cp, eeP)
        dpick = int(np.argmin(dline))
        nodewN = node[np.array(sorted(have))]
        _, dn = _dsearchn(nodewN, eeP[dpick][None, :])
        dnode = nodewN[int(np.argmin(spdist(eeP[dpick], nodewN)))]
        d0 = int(np.flatnonzero(np.all(np.isclose(node, dnode), axis=1))[0])

        Dist = squareform(pdist(node))
        adj = nx.to_numpy_array(G, nodelist=range(node.shape[0]))
        adj = (adj > 0).astype(float)
        AdjMax = adj * Dist
        ee_arg = remNode.reshape(-1, 1) if remNode.size else None
        Path, _, add, st = ChinesePostman(adj, AdjMax, Dist, [], ee_arg, d0)
        Path = list(np.atleast_1d(Path).astype(int))
    else:
        d0 = int(np.argmin(spdist(cp, node)))
        el0 = edgeList[0]
        Path = [int(el0[d0])] + [int(x) for x in el0 if int(x) != int(el0[d0])]

    waypoint = node[Path]
    waypoint_coords = waypoint.copy()

    # ---- assemble fill waypoints with transit and fill flags ----
    _, d1 = _dsearchn(waypoint_coords[0][None, :], realEndP)
    _, d2 = _dsearchn(waypoint_coords[-1][None, :], realEndP)
    d1 = (d1 <= a + 5).any()
    d2 = (d2 <= a + 5).any()
    endPP = np.array([[waypoint_coords[0, 0], waypoint_coords[0, 1], float(d1)],
                      [waypoint_coords[-1, 0], waypoint_coords[-1, 1], float(d2)]])

    if endPP[:, 2].any():
        flag = 1
        if endPP[0, 2] != endPP[1, 2]:
            do_flip = bool(endPP[1, 2])
        else:
            do_flip = (int(np.argmin(spdist(cp, endPP[:, [0, 1]]))) == 1)
        if do_flip:
            waypoint_coords = waypoint_coords[::-1]
        waypoint_coords = _assemble_waypoints(waypoint_coords, cp, crackRaw, a)
    else:
        flag = 0
        pass   # waypoint_coords left as-is; caller ignores waypoints when flag=0

    return waypoint_coords, flag


def _assemble_waypoints(waypoint_coords, cp, crackRaw, a):
    """Assemble the final waypoint sequence with transit and fill flags.

    Interpolates transit waypoints from the robot's current position to the
    first fill waypoint, then interpolates between consecutive fill waypoints.
    Each interpolated point is flagged 1 (nozzle on) if it lies within
    ``a * sqrt(2)`` pixels of any crack pixel, or 0 (transit) otherwise.

    Parameters
    ----------
    waypoint_coords : ndarray, shape (W, 2)
        Ordered fill waypoints from the Chinese-Postman route.
    cp : ndarray, shape (2,)
        Robot's current position.
    crackRaw : list of ndarray
        Crack polylines used to assign fill flags.
    a : int
        Nozzle footprint radius in pixels.

    Returns
    -------
    out : ndarray, shape (N, 3)
        Waypoints with columns ``[row, col, fill_flag]``.
    """
    mx1, my1 = addPtsLin([cp[0], waypoint_coords[0, 0]], [cp[1], waypoint_coords[0, 1]], a + 5)
    mx2 = []
    my2 = []
    for h in range(waypoint_coords.shape[0] - 1):
        m, n = addPtsLin(waypoint_coords[[h, h + 1], 0], waypoint_coords[[h, h + 1], 1], a)
        mx2.extend([waypoint_coords[h, 0]] + list(m))
        my2.extend([waypoint_coords[h, 1]] + list(n))
    mx2 = np.asarray(mx2, float)
    my2 = np.asarray(my2, float)
    b = [1.0] if mx1.size else []
    dd = np.zeros(mx2.shape[0], dtype=bool)
    for cr in crackRaw:
        _, t = _dsearchn(np.asarray(cr, float), np.column_stack([mx2, my2]))
        dd = dd | (t <= a * sqrt(2))

    block1_flags = np.concatenate([np.zeros(max(mx1.size - 1, 0)), np.array(b)]) if mx1.size else np.empty(0)
    block1 = np.column_stack([mx1, my1, block1_flags]) if mx1.size else np.empty((0, 3))
    block2 = np.column_stack([mx2, my2, dd.astype(float)])
    last = np.array([[waypoint_coords[-1, 0], waypoint_coords[-1, 1], 1.0]])
    return np.vstack([block1, block2, last])


def _intersect_nonempty(aRan, c):
    """Return True if polygon ``aRan`` and polyline ``c`` intersect.

    Parameters
    ----------
    aRan : PolyShape
        Buffer polygon (nozzle footprint disk).
    c : array-like, shape (M, 2)
        Crack polyline vertices.

    Returns
    -------
    bool
        True if they intersect; False if either is empty or degenerate.
    """
    g = pu._as_geom(aRan)
    if g is None or g.is_empty:
        return False
    from shapely.geometry import LineString
    cr = np.atleast_2d(np.asarray(c, dtype=float))
    if cr.shape[0] < 2:
        return False
    return LineString(cr).intersects(g)

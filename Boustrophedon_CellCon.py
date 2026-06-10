"""Multi-cell boustrophedon coverage with cell-connection optimisation.

:func:`Boustrophedon_CellCon` extends the single-region boustrophedon planner to
handle a decomposition that contains multiple, possibly disjoint, coverage cells.
It solves a combined orientation-and-ordering problem before sweeping:

1. **Pre-computation** — for each Reeb-edge cell, :func:`BoustrophedonPath` is
   called for both sweep directions (left-to-right and right-to-left) to obtain
   each direction's start/end (x, y) coordinates and total sweep length.
2. **Cell-connection graph** — a directed graph is built whose nodes are the
   cell entry and exit points.  Within-cell edges are weighted by sweep length;
   between-cell connection edges are weighted by Euclidean distance.
3. **Shortest-path optimisation** — Dijkstra's algorithm finds the minimum-cost
   path from the first cell's entry to the last cell's exit over the four
   possible (first-cell start direction, last-cell end direction) combinations.
   The chosen path determines the per-cell sweep direction ``se`` and the
   optimised entry points ``seP``.
4. **Main sweep** — each cell is swept via :func:`BoustrophedonPath` in the
   chosen order and direction; consecutive cell paths are linked with
   intermediate waypoints at sensor spacing ``s``.

This is the primary coverage driver for the OnlineSCC and SCC planners when the
decomposition produces more than one cell.
"""

import math

import numpy as np

from private import poly_utils as pu
from private.utils import addPtsLin, spdist2, total_length
from BoustrophedonPath import BoustrophedonPath

_XMAX, _YMAX = 3048, 2898

# Last cell-connection decision captured for diagnostic use (ccPath/l/seP).
_LAST = {}


def _rmmissing(a):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    return a[~np.isnan(a).any(axis=1)]


def _shortestpath(adj, src, dst, n):
    """Dijkstra shortest path with deterministic tie-breaking.

    Finds the minimum-cost path from ``src`` to ``dst`` in the directed graph
    ``adj``.  Tie-breaking rules:

    - Neighbours are explored in ascending target-node order.
    - Priority-queue ties are broken by the lower node index.
    - On an equal-cost relaxation, the lower-indexed predecessor is kept.

    These rules ensure deterministic ordering on the cell-connection graph.

    Parameters
    ----------
    adj : dict
        Adjacency mapping ``{node: [(neighbour, weight), ...]}``.
        Neighbour lists must already be sorted in ascending order.
    src : int
        Source node (1-based).
    dst : int
        Destination node (1-based).
    n : int
        Total number of nodes in the graph.

    Returns
    -------
    path : list of int or None
        Ordered node sequence from ``src`` to ``dst``, or ``None`` if
        unreachable.
    dist : float
        Total path cost, or ``math.inf`` if unreachable.
    """
    import heapq
    INF = math.inf
    dist = {v: INF for v in range(1, n + 1)}
    pred = {v: 0 for v in range(1, n + 1)}      # 0 = no predecessor (1-based ids)
    dist[src] = 0.0
    heap = [(0.0, src)]
    done = set()
    while heap:
        d_u, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == dst:
            break
        for v, w in adj.get(u, ()):             # already ascending by v
            if v in done:
                continue
            nc = dist[u] + w
            if nc < dist[v] or (nc == dist[v] and (pred[v] == 0 or u < pred[v])):
                dist[v] = nc
                pred[v] = u
                heapq.heappush(heap, (nc, v))
    if dist[dst] == INF:
        return None, INF
    path = [dst]
    while path[-1] != src:
        path.append(pred[path[-1]])
    path.reverse()
    return path, dist[dst]


def _edge_index(reebT, EE):
    """Return the first index in ``reebT`` that matches ``EE`` in either orientation, or ``None``."""
    for k, e in enumerate(reebT):
        if (e[0] == EE[0] and e[1] == EE[1]) or (e[0] == EE[1] and e[1] == EE[0]):
            return k
    return None


def _is_reeb_edge(EE, reebEdge):
    return _edge_index(reebEdge, EE) is not None


def _clamp(subXY, sensor):
    subXY = np.atleast_2d(np.array(subXY, dtype=float))
    subXY[subXY[:, 0] < sensor, 0] = sensor
    subXY[subXY[:, 0] > _XMAX - sensor, 0] = _XMAX - sensor
    subXY[subXY[:, 1] < sensor, 1] = sensor
    subXY[subXY[:, 1] > _YMAX - sensor, 1] = _YMAX - sensor
    return subXY


def Boustrophedon_CellCon(Path, splitReg, see, seP, init, wall_fol, known, sim,
                          reebEdge, reebCell, allNode, s, a, crack_reach=None, n_crack=0):
    """Generate the boustrophedon coverage path for multiple cells with connection optimisation.

    Sweeps all coverage cells in the decomposition and connects them with
    minimum-cost links.  Before sweeping, the function optimises the per-cell
    sweep direction and entry point by solving a shortest-path problem over a
    cell-connection directed graph (see module docstring for the full procedure).

    Parameters
    ----------
    Path : array-like, shape (N, 2)
        Reeb traversal order — each row ``[start, end]`` is a pair of
        critical-point node indices defining one cell or connection edge.
    splitReg : list
        Coverage cells from the Morse Cell Decomposition (MCD), indexed by
        the entries in ``reebCell``.
    see : ignored
        Legacy input; sweep directions are recomputed internally.
    seP : ignored
        Legacy input; entry points are recomputed internally.
    init : array-like, shape (2,)
        Initial robot position (x, y).  Used when ``known`` is ``False`` to
        add the travel cost from the current position to the first cell.
    wall_fol : array-like, shape (N,)
        Per-edge wall-follow flag (1 = wall-follow prefix required, 0 = plain
        sweep).
    known : bool
        ``True`` for the offline (SCC) planner; ``False`` for the online
        (OnlineSCC) planner.
    sim : bool
        When ``True``, disables boundary clamping (simulation mode).
    reebEdge : array-like, shape (E, 2)
        Reeb graph edges as pairs of critical-point node indices.
    reebCell : array-like, shape (E,)
        Index into ``splitReg`` for each Reeb edge.
    allNode : array-like, shape (M, 2)
        Critical-point coordinates in (row, col) order as produced by MCD.
    s : float
        Sensor radius in pixels.  Sweep spacing is ``s / sqrt(2)``; inter-cell
        link waypoints are placed at spacing ``s``.
    a : float
        Robot footprint radius in pixels.
    crack_reach : float or None, optional
        Wall-clamp distance (pixels) applied when driving to a genuine crack-fill
        node near a wall.  When ``None`` (default, used by OnlineSCC) all
        waypoints are clamped to the sensor inset ``s / sqrt(2)``.
    n_crack : int, optional
        Number of crack-graph nodes at the front of ``allNode``.  Only edges
        whose both endpoints are crack nodes (index < ``n_crack``) use
        ``crack_reach`` for clamping; all other edges keep the sensor inset.

    Returns
    -------
    PathEdge : ndarray, shape (K, 2)
        Ordered (x, y) coverage waypoints for the entire decomposition.
    """
    # crack_reach: optional wall-clamp bound (px) for crack-fill edges only.
    # The cell sweep is always clamped to the sensor inset s/sqrt2 (the inscribed-
    # square coverage guarantee). A crack near a wall may be reached by the footprint
    # down to the base-reach limit; SCC passes crack_reach so near-wall crack nodes
    # are not dragged in to the sensor inset. n_crack = number of crack-graph nodes
    # (the first n_crack rows of allNode); only an edge whose both endpoints are crack
    # nodes (index < n_crack) gets crack_reach. Default crack_reach=None -> clamp to
    # sensor everywhere (OnlineSCC behaviour).
    reebEdge = np.atleast_2d(np.asarray(reebEdge, dtype=int)).copy()
    reebCell = np.asarray(reebCell, dtype=int)
    allNode = np.atleast_2d(np.asarray(allNode, dtype=float))
    Path = np.atleast_2d(np.asarray(Path, dtype=int))
    wall_fol = np.atleast_1d(np.asarray(wall_fol, dtype=int))
    sensor = s / math.sqrt(2)
    fl_see = False

    init_arr = np.atleast_1d(np.asarray(init, dtype=float))
    has_init = init_arr.size > 1 and np.all(init_arr != 0)
    PathEdge = init_arr[None, :] if has_init else np.empty((0, 2))

    # ----- cell connection: collect entry/exit nodes for both sweep directions -----
    reebT = reebEdge.copy()
    ccNode = []
    ccW = []
    for i in range(Path.shape[0]):
        EE = Path[i, :2]
        ind = _edge_index(reebT, EE)
        if ind is not None:
            cell = splitReg[int(reebCell[ind])]
            reebT[ind] = [0, 0]
            orgcell = cell
            if not wall_fol[i]:
                init_wf = []
            else:
                init_wf = allNode[int(EE[0])][::-1]      # (row,col) -> (x,y)
            for dr in (0, 1):
                subXY, _ = BoustrophedonPath(cell, orgcell, EE, sensor, dr, init_wf,
                                             int(wall_fol[i]), known, fl_see, allNode, s, a)
                subXY = np.atleast_2d(subXY)
                if subXY.size:
                    ccNode.append(subXY[0])
                    ccNode.append(subXY[-1])
                    ccW.append(total_length(_rmmissing(subXY)))
        else:
            sub = np.array([allNode[int(EE[0])][::-1], allNode[int(EE[1])][::-1]])
            tl = total_length(_rmmissing(sub))
            ccNode.extend([sub[0], sub[-1], sub[0], sub[-1]])
            ccW.extend([tl, tl])

    ccNode = np.array(ccNode, dtype=float)
    ccW = np.array(ccW, dtype=float)
    M = ccNode.shape[0]

    # within-cell (start->end) edges
    i_within = np.arange(1, M + 1, 2)                     # 1,3,5,...
    ccEdge = np.column_stack([i_within, i_within + 1])
    # direction label per start node: 4c-3 -> dir0, 4c-1 -> dir1
    sDir = np.tile([0, 1], len(i_within))[:len(i_within)]
    # between-cell connection edges
    i_conn = np.arange(2, M - 3 + 1, 4)                   # 2,6,10,...
    if i_conn.size:
        conn = np.vstack([
            np.column_stack([i_conn, i_conn + 3]),
            np.column_stack([i_conn, i_conn + 5]),
            np.column_stack([i_conn + 2, i_conn + 3]),
            np.column_stack([i_conn + 2, i_conn + 5]),
        ])
        ccEdge = np.vstack([ccEdge, conn])

    # weights: connection edges by Euclidean distance; within-cell edges by sweep length
    ccWeight = spdist2(ccNode[ccEdge[:, 0] - 1], ccNode[ccEdge[:, 1] - 1])
    ccWeight[:len(ccW)] = ccW

    # adjacency in ascending target-node order for deterministic Dijkstra tie-breaking
    adj = {}
    for (u, v), w in zip(ccEdge, ccWeight):
        adj.setdefault(int(u), []).append((int(v), float(w)))
    for u in adj:
        adj[u].sort(key=lambda vw: vw[0])

    def _sp(src, dst):
        return _shortestpath(adj, src, dst, M)

    combos = [(1, M), (1, M - 2), (3, M), (3, M - 2)]
    paths, dists = [], []
    for src, dst in combos:
        p, d = _sp(src, dst)
        if not known and p is not None:
            d = d + spdist2(np.atleast_2d(init), ccNode[src - 1:src])[0]
        paths.append(p)
        dists.append(d)
    l = int(np.argmin(dists))
    ccPath = np.array(paths[l][0::2])                     # every other node = cell start nodes

    # which Path rows are Reeb edges (either orientation)
    reeb_mask = np.array([_is_reeb_edge(Path[r, :2], reebEdge) for r in range(Path.shape[0])])
    sel_within = np.isin(i_within, ccPath)
    se = sDir[sel_within][reeb_mask]
    seP = ccNode[ccPath - 1][reeb_mask]

    _LAST.clear()
    _LAST.update(ccPath=ccPath.tolist(), l=l + 1, dists=[float(d) for d in dists],
                 paths=[(list(map(int, p)) if p is not None else None) for p in paths],
                 M=int(M), se=se.tolist(), seP=seP.tolist(), ccNode=ccNode,
                 ccEdge=ccEdge.copy(), ccWeight=ccWeight.copy(),
                 n_within=int(len(ccW)))

    # ----- main pass: sweep each cell in chosen order/direction, link cells -----
    reebEdge_w = reebEdge.copy()
    o = -1
    for i in range(Path.shape[0]):
        EE = Path[i, :2]
        ind = _edge_index(reebEdge_w, EE)
        if ind is not None:
            o += 1
            cell = splitReg[int(reebCell[ind])]
            reebEdge_w[ind] = [0, 0]
            orgcell = cell
            subXY, _ = BoustrophedonPath(cell, orgcell, EE, sensor, int(se[o]), init,
                                         int(wall_fol[i]), known, fl_see, allNode, s, a)
            subXY = np.atleast_2d(subXY)
            if not sim:
                subXY = _clamp(subXY, sensor)
        else:
            if PathEdge.shape[0] == 0:
                PathEdge = allNode[int(EE[0])][::-1][None, :]
            nxt_reeb = (i != Path.shape[0] - 1) and _is_reeb_edge(Path[i + 1, :2], reebEdge)
            if nxt_reeb and seP.shape[0] > o + 1:
                mx, my = addPtsLin([PathEdge[-1, 0], seP[o + 1, 0]],
                                   [PathEdge[-1, 1], seP[o + 1, 1]], s)
                link = np.column_stack([mx, my]) if len(mx) else np.empty((0, 2))
                subXY = np.vstack([PathEdge[-1], link, seP[o + 1]])
            else:
                if known:
                    subXY = np.vstack([PathEdge[-1], allNode[int(EE[1])][::-1]])
                else:
                    subXY = PathEdge[-1:].copy()
            if not sim:
                # Apply crack_reach only when driving to a genuine crack node
                # (target index < n_crack); transit links to cell critical points
                # on the wall keep the sensor clamp to avoid unnecessary wall pokes.
                to_crack = crack_reach is not None and int(EE[1]) < n_crack
                subXY = _clamp(subXY, crack_reach if to_crack else sensor)

        if subXY is not None and len(subXY):
            subXY = np.atleast_2d(subXY)
            if PathEdge.shape[0] and not known:
                mx, my = addPtsLin([PathEdge[-1, 0], subXY[0, 0]],
                                   [PathEdge[-1, 1], subXY[0, 1]], s)
                link = np.column_stack([mx, my]) if len(mx) else np.empty((0, 2))
                PathEdge = np.vstack([PathEdge, link, subXY])
            else:
                PathEdge = np.vstack([PathEdge, subXY])
            init = subXY[-1]

    return PathEdge

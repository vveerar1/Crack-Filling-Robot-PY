"""Reeb-graph cell-order traversal for boustrophedon coverage planning.

This module provides two functions that determine the order in which the robot
visits the coverage cells identified by :func:`Reeb`:

:func:`ReebPath`
    Recursive depth-first traversal of the Reeb graph starting from a given
    critical point.  At each node the reachable neighbours are sorted by their
    connectivity (how many onward edges remain) and then by cell area, so that
    the traversal prefers to continue into better-connected subtrees before
    visiting dead-end branches.  Neighbours to the left of the current node
    are visited before neighbours to the right within each connectivity group.
    Returns the cell visit sequence (*Path*) and a flag per step indicating
    whether that cell is a dead-end requiring wall-following (*wall_fol*).

:func:`reeb_traversal`
    Master traversal that calls :func:`ReebPath` from the initial start node
    and then, if any Reeb edges remain uncovered (because the graph is
    disconnected), jumps to the nearest unvisited critical point and continues
    until every edge has been visited.  This stitch-and-continue strategy
    guarantees complete coverage even when the free space consists of multiple
    disconnected regions.

Path format
-----------
Each row of *Path* is ``[from_node, to_node, cell_index]``, where the indices
are 0-based.  A *cell_index* of ``-1`` marks a jump between disconnected
components (no actual sweep cell is traversed on that step).

The adjacency matrix *adj* is passed by value into each recursive call and
mutated in place so that each edge is consumed exactly once.
"""

import numpy as np
from private.utils import spdist

_NOCELL = -1

# Per-call connectivity debug log.  Set ReebPath._DBG_ON = True before a run
# to capture neighbour-selection rows; disabled by default to avoid unbounded growth.
_DBG = []
_DBG_ON = False


def _sortrows_by_area_asc(rows):
    """rows: list of [neighbor, con, left, cellOrd, area]; stable sort by area asc."""
    return sorted(rows, key=lambda r: r[4])


def ReebPath(adj, critP, reebEdge, cells, Start):
    """Recursively traverse the Reeb graph from a given start node.

    Performs a depth-first traversal of the cell-adjacency graph, consuming
    edges as they are visited.  Neighbours are sorted first by their remaining
    connectivity (ascending), then by cell area (ascending), with left-of-current
    neighbours placed before right-of-current neighbours in each group.  This
    ordering keeps the traversal path spatially coherent and avoids unnecessary
    backtracking.

    Parameters
    ----------
    adj : array-like, shape (N, N)
        Reeb-graph adjacency matrix.  Entry ``adj[i, j]`` is the number of edges
        between critical points *i* and *j*.  Modified in place as edges are
        consumed; a copy is made at entry so the caller's array is not affected.
    critP : array-like, shape (N, 2)
        Critical-point coordinates in ``(y, x)`` order, used to determine whether
        each neighbour is to the left or right of the current node.
    reebEdge : array-like, shape (E, 2)
        Reeb edge list (pairs of critical-point indices) from :func:`Reeb`.  Used
        to look up which cell index corresponds to each traversed edge.
    cells : list of PolyShape
        Coverage cells from :func:`MCD`, indexed by the cell column of *reebEdge*.
        Their areas are used to break traversal ties.
    Start : int
        0-based index of the critical point from which traversal begins.

    Returns
    -------
    Path : list of [int, int, int]
        Ordered traversal steps, each ``[from_node, to_node, cell_index]``.
        *cell_index* is the 0-based index into *cells* for the swept cell, or
        ``-1`` if the edge does not correspond to a cell.
    wall_fol : list of int
        Per-step wall-following flag: ``1`` if the destination node is a dead-end
        (all its remaining edges were just consumed) and the cell should be swept
        with a wall-following pass; ``0`` otherwise.  The last entry is always 0.
    adj : ndarray
        The adjacency matrix after all edges reachable from *Start* have been
        consumed.
    """
    adj = np.array(adj, dtype=float)        # work on a copy we mutate + return
    critP = np.atleast_2d(np.asarray(critP, dtype=float))
    reebEdge = np.atleast_2d(np.asarray(reebEdge, dtype=int))
    ncells = len(cells)

    Path = []
    wall_fol = []
    ind = int(Start)

    no = np.flatnonzero(adj[ind, :] > 0)
    repeat = adj[ind, no].astype(int)
    neighbor = []
    for j in range(len(no)):
        neighbor.extend([int(no[j])] * int(repeat[j]))

    # cellOrd: edge (cell) index for each neighbor edge, consuming matches
    cellOrd = []
    treebE = reebEdge.copy()
    for nb in neighbor:
        m = np.flatnonzero((treebE[:, 0] == ind) & (treebE[:, 1] == nb))
        if m.size:
            lib = int(m[0]); treebE[lib] = [-99, -99]
            cellOrd.append(lib if lib < ncells else _NOCELL)
        m = np.flatnonzero((treebE[:, 0] == nb) & (treebE[:, 1] == ind))
        if m.size:
            lib = int(m[0]); treebE[lib] = [-99, -99]
            cellOrd.append(lib if lib < ncells else _NOCELL)

    neighbor = np.array(neighbor, dtype=int)
    cellOrd = np.array(cellOrd, dtype=int)

    # connectivity of each neighbor excluding the edge back to ind
    cols = np.arange(adj.shape[1]) != ind
    neighborCon = np.array([adj[nb, cols].sum() for nb in neighbor])
    leftcell = (critP[neighbor, 1] < critP[ind, 1]).astype(int)

    # Guard: _NOCELL=-1 means "no cell assigned"; cell index 0 is a valid cell.
    # Use != _NOCELL (not != 0) to avoid zeroing areas when cell 0 is a neighbour.
    areas = np.array([cells[c].area if c != _NOCELL else 0.0 for c in cellOrd]) \
        if (cellOrd != _NOCELL).all() else np.zeros(len(neighbor))

    # conn rows: [neighbor, con, left, cellOrd, area]
    conn = [[int(neighbor[k]), float(neighborCon[k]), int(leftcell[k]),
             int(cellOrd[k]), float(areas[k])] for k in range(len(neighbor))]
    if _DBG_ON:
        _DBG.append({"ind": int(ind), "ind_xy": critP[ind].tolist(),
                     "conn": [r[:] for r in conn],
                     "nbr_xy": {int(neighbor[k]): critP[int(neighbor[k])].tolist() for k in range(len(neighbor))}})

    adj[ind, :] = 0
    adj[:, ind] = 0

    for i in sorted(set(r[1] for r in conn)):
        group = [r for r in conn if r[1] == i]
        left = _sortrows_by_area_asc([r for r in group if r[2]])
        right = _sortrows_by_area_asc([r for r in group if not r[2]])
        temp2 = left + right
        for r in temp2:
            Path.append([ind, r[0], r[3]])
            wall_fol.append(1 if r[1] == 0 else 0)
        if i > 0:
            start_j = temp2[0][0]
            tPath, twall, adj = ReebPath(adj, critP, reebEdge, cells, start_j)
            Path.extend(tPath)
            wall_fol.extend(twall)

    if Path and adj[:, Path[-1][1]].sum() > 0:
        tPath, twall, adj = ReebPath(adj, critP, reebEdge, cells, Path[-1][1])
        Path.extend(tPath)
        wall_fol.extend(twall)

    if adj.sum() == 0 and wall_fol:
        wall_fol[-1] = 0

    return Path, wall_fol, adj


def _covered_count(reebEdge, Path):
    """Number of reebEdge rows present in Path[:, :2] (either orientation)."""
    if not Path:
        return 0
    P = np.array(Path, dtype=int)[:, :2]
    pset = set(map(tuple, P)) | set(map(tuple, P[:, ::-1]))
    return sum(1 for e in reebEdge if tuple(e) in pset)


def reeb_traversal(adj, critP, reebEdge, cells, Start):
    """Plan a complete cell visit sequence over the full Reeb graph.

    Calls :func:`ReebPath` from *Start* to produce an initial traversal, then
    repeatedly detects uncovered Reeb edges (due to disconnected free-space
    regions), jumps to the nearest unvisited critical point, and continues the
    traversal until every edge has been visited.  Each inter-component jump is
    recorded in *Path* as a step with ``cell_index = -1`` to indicate that no
    sweep cell is being traversed.

    Parameters
    ----------
    adj : array-like, shape (N, N)
        Reeb-graph adjacency matrix (see :func:`ReebPath`).
    critP : array-like, shape (N, 2)
        Critical-point coordinates in ``(y, x)`` order.
    reebEdge : array-like, shape (E, 2)
        Reeb edge list from :func:`Reeb`.
    cells : list of PolyShape
        Coverage cells from :func:`MCD`.
    Start : int
        0-based index of the critical point at which to begin traversal.

    Returns
    -------
    Path : list of [int, int, int]
        Complete ordered traversal covering every Reeb edge.  Format is the same
        as :func:`ReebPath`: ``[from_node, to_node, cell_index]`` per step.
    wall_fol : list of int
        Wall-following flag for each step (see :func:`ReebPath`).
    adj : ndarray
        Adjacency matrix after all edges have been consumed (should be all zeros).
    """
    critP = np.atleast_2d(np.asarray(critP, dtype=float))
    reebEdge = np.atleast_2d(np.asarray(reebEdge, dtype=int))

    Path, wall_fol, adj = ReebPath(adj, critP, reebEdge, cells, Start)
    if wall_fol:
        wall_fol[-1] = 0

    while _covered_count(reebEdge, Path) != reebEdge.shape[0]:
        Start = Path[-1][1]
        visited = set(int(v) for row in Path for v in row[:2])
        t = np.array([n for n in range(critP.shape[0]) if n not in visited], dtype=int)
        if t.size == 0:
            break
        ind = int(np.argmin(spdist(critP[Start], critP[t])))
        nxt = int(t[ind])
        Path.append([Start, nxt, -1])   # -1 = jump between disconnected components
        wall_fol.append(0)
        Start = nxt
        tPath, twall, adj = ReebPath(adj, critP, reebEdge, cells, Start)
        Path.extend(tPath)
        wall_fol.extend(twall)
        if wall_fol:
            wall_fol[-1] = 0

    return Path, wall_fol, adj

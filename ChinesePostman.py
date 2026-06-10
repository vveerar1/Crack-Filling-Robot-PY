"""Modified Chinese Postman / Rural Postman route optimiser.

The Chinese Postman Problem (CPP) asks for the shortest closed walk that
traverses every edge of a graph at least once.  This module solves a modified
version used to order the robot's coverage cells and crack-fill edges:

- **Eulerian graph** (all node degrees even) — Fleury's algorithm produces the
  circuit directly.
- **Non-Eulerian graph** — the minimum-weight set of extra edges needed to make
  all node degrees even is found by solving a binary Integer Linear Program (ILP)
  over candidate odd-degree node pairs, using ``scipy.optimize.milp``.  The graph
  is augmented with those edges and Fleury's algorithm is then applied.

If the augmented graph is still disconnected (e.g. isolated crack components),
a minimum spanning tree reconnects the components before re-matching.

:func:`ChinesePostman` is the main entry point.  It returns the node traversal
order (``Path``), total path weight (``Weight``), the added matching edges
(``add``), and the starting edge (``st``).
"""

import numpy as np
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds

# When True, reconnection links are restricted to odd-degree nodes (crack endpoints
# and junctions) so that a reconnect edge never lands on a mid-crack degree-2
# pass-through node.  A component-spanning fallback reverts to all edge-nodes when
# no odd node can reach across a component.  Set False to use all edge-nodes.
_RECONNECT_ODD_ONLY = True


def _combnk_pairs(v):
    """Return all 2-element combinations of ``v`` in a specific deterministic order.

    Pairs are emitted with the larger element descending as the outer key and
    the smaller element descending as the inner key — e.g.
    ``[1,3,4,5]`` yields ``(4,5),(3,5),(1,5),(3,4),(1,4),(1,3)``.
    This order determines the matched-edge list order, which in turn sets the
    Fleury start-node tie-break.
    """
    v = list(v)
    n = len(v)
    pairs = [(v[i], v[j]) for j in range(n - 1, 0, -1) for i in range(j - 1, -1, -1)]
    return np.array(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)


def _bin(A):
    return (np.asarray(A) > 0).astype(int)


def _euler_test(A):
    A = np.asarray(A, dtype=float)
    B = _bin(A)
    deg = B.sum(axis=1)
    odd = int(np.sum(deg % 2 != 0))
    if odd == 0:
        return 1
    if odd == 2:
        return 2
    return 3


def _connected_active(B):
    """Return ``True`` if the subgraph induced by non-isolated vertices of ``B`` is connected."""
    nz = np.flatnonzero((B.sum(axis=1) + B.sum(axis=0)) > 0)
    if nz.size == 0:
        return True
    G = nx.Graph()
    G.add_nodes_from(nz.tolist())
    bb = _bin(B)
    for i in nz:
        for j in nz:
            if j > i and bb[i, j]:
                G.add_edge(int(i), int(j))
    return nx.is_connected(G.subgraph(nz.tolist()))


def _fleury_conn(A, R, C, T):
    """Return 1 if traversing edge ``(R, C)`` keeps the graph connected, else 0."""
    if C == R and A.sum() != 2 and A[C, :].sum() == 1:
        return 0
    B = A.astype(float).copy()
    if not T:
        B[R, C] -= 1
        B[C, R] -= 1
    return 1 if _connected_active(B) else 0


def _fleury_row(A, R, P=None):
    """Perform one Fleury step from node ``R``, returning ``(A, R, C, P)``.

    Selects a non-bridge edge ``(R, C)`` to traverse, decrements its count in
    ``A``, and returns the updated adjacency matrix along with the new current
    node ``C``.  When ``P`` is provided (the matched-edge list from odd-node
    pairing), the highest-count edge is preferred; otherwise columns are scanned
    in order.
    """
    A = A.astype(float).copy()
    n = A.shape[0]
    if P is not None:                       # shortest-path-aware variant (case 5)
        lP = [list(e) for e in P]
        m = int(np.max(A[R, :]))
        con = []
        for p in range(m, 0, -1):
            con = list(np.flatnonzero(A[R, :] == p))
            if con:
                break
        for C in con:
            if A[R, C] > 0:
                T = 0
                ind = -1
                for h, (t1, t2) in enumerate(lP):
                    if (R == t2 and C == t1) or (C == t2 and R == t1):
                        T = 1
                        ind = h
                        lP[h] = [0, 0]
                        break
                Cval = _fleury_conn(A, R, C, T)
                if Cval == 0 and ind != -1:
                    lP[ind] = [R, C]
                if Cval == 1 and (A[:, C].sum() > 1 or A.sum() == 2):
                    if T == 0:
                        A[R, C] -= 1
                        A[C, R] -= 1
                    return A, R, C, np.array(lP)
        # When no candidate is immediately traversable (all are temporary bridges),
        # advance R to the last/highest-index highest-count neighbour without
        # consuming an edge.
        return A, R, (int(con[-1]) if len(con) else R), np.array(lP)
    else:                                   # plain Eulerian variant (case 2)
        for C in range(n):
            if A[R, C] > 0:
                if _fleury_conn(A, R, C, 0) == 1:
                    A[R, C] -= 1
                    A[C, R] -= 1
                    return A, R, C
        # When no traversable edge exists (all are temporary bridges),
        # advance R to the last node without consuming an edge.
        return A, R, n - 1


def _fleury(A, ADJ, start, P=None, ow=0.0):
    A = np.asarray(A, dtype=float)
    w = A.sum() / 2.0 if np.allclose(A, A.T) else A.sum()
    if P is not None:
        w += ow
    M = np.asarray(ADJ, dtype=float).copy()
    x = []
    R = int(start)
    Pl = None if P is None else np.asarray(P)
    while M.sum() != 0:
        if P is not None:
            M, R, C, Pl = _fleury_row(M, R, Pl)
        else:
            M, R, C = _fleury_row(M, R)
        x.append(R)
        R = C
    x.append(C)
    return x, w


def _linear_prog(G_edges, b_n, DIST, ee, flag, rmflag):
    """Find the minimum-weight matching of odd-degree nodes via binary ILP.

    Solves a binary Integer Linear Program (ILP) over all candidate odd-node
    pairs to select the minimum-cost set of extra edges that makes every node
    degree even (a necessary condition for an Euler circuit).

    Parameters
    ----------
    G_edges : list of (int, int)
        Existing edges in the graph (used to optionally exclude them from
        candidacy when ``rmflag`` is ``True``).
    b_n : ndarray, shape (N,)
        Binary indicator: ``b_n[i] == 1`` if node ``i`` has odd degree.
    DIST : ndarray, shape (N, N)
        All-pairs shortest-path distance matrix.
    ee : array-like or None
        Additional edges to exclude from candidacy (e.g. already-covered edges).
    flag : bool
        Reserved; currently unused.
    rmflag : bool
        When ``True``, existing graph edges are removed from the candidate pairs.

    Returns
    -------
    sel : ndarray, shape (K, 2)
        Selected matching pairs ``[u, v]`` (node indices), or an empty array if
        no pairs exist or the ILP is infeasible.
    """
    odd = np.flatnonzero(b_n)
    comb = _combnk_pairs(odd.tolist())
    if comb.size == 0:
        return np.empty((0, 2), dtype=int)
    if rmflag:
        eset = {tuple(sorted(e)) for e in G_edges}
        keep = [i for i, e in enumerate(comb) if tuple(sorted(e)) not in eset]
        comb2 = comb[keep]
        if np.unique(comb2).size == odd.size:    # keep only if still covers all odd nodes
            comb = comb2
    if ee is not None and np.ndim(ee) == 2 and len(ee):
        eset = {tuple(sorted(e)) for e in np.atleast_2d(ee)}
        comb = np.array([e for e in comb if tuple(sorted(e)) not in eset])
    c_e = DIST[comb[:, 0], comb[:, 1]]

    n_pair = comb.shape[0]
    nodes = np.unique(comb)
    a_ne = np.zeros((nodes.size, n_pair))
    node_idx = {int(v): k for k, v in enumerate(nodes)}
    for i, (u, v) in enumerate(comb):
        a_ne[node_idx[int(u)], i] = 1
        a_ne[node_idx[int(v)], i] = 1
    b = np.array([1.0 if b_n[int(v)] else 0.0 for v in nodes])

    res = milp(c=c_e, constraints=LinearConstraint(a_ne, b, b),
               integrality=np.ones(n_pair), bounds=Bounds(0, 1))
    if not res.success or res.x is None:
        return np.empty((0, 2), dtype=int)
    sel = comb[res.x > 0.5]
    return sel


def _opt_odd_pair(A, DIST, ee, cl, flag, rmflag):
    # Odd-degree test uses the COUNT (multigraph) degree: parallel edges each
    # contribute to degree, so two parallel edges between a node pair make both
    # endpoints odd and they must be matched before Fleury can complete.
    b_n = (np.asarray(A, dtype=float).sum(axis=1) % 2 != 0).astype(int)
    G_edges = [(i, j) for i in range(A.shape[0]) for j in range(i + 1, A.shape[0]) if A[i, j] > 0]
    if not b_n.any():
        return np.empty((0, 2), dtype=int), 0.0, np.empty((0, 2), dtype=int)
    added = _linear_prog(G_edges, b_n, DIST, ee, flag, rmflag)
    st = np.empty((0, 2), dtype=int)
    if added.size:
        s = int(np.argmax(DIST[added[:, 0], added[:, 1]]))
        st = added[s:s + 1]
    weight = float(np.sum(DIST[added[:, 0], added[:, 1]])) if added.size else 0.0
    return added, weight, st


def _circuit_reconnect(A, ADJ, DIST, added):
    """Reconnect a disconnected augmented graph using a minimum spanning tree.

    After odd-node matching, if the resulting graph (original edges + matched
    edges) is still disconnected (e.g. isolated crack sub-graphs), this function
    adds the minimum extra edges needed to span all components.  A minimum
    spanning tree is built over candidate node pairs, with existing edges given
    zero weight so that the tree retains them and only adds true cross-component
    links.

    The adjacency matrices ``A`` and ``ADJ`` are modified in place with the new
    edges.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Weighted adjacency matrix of the original graph (modified in place).
    ADJ : ndarray, shape (N, N)
        Edge-count (multiplicity) adjacency matrix (modified in place).
    DIST : ndarray, shape (N, N)
        All-pairs shortest-path distance matrix.
    added : ndarray, shape (K, 2)
        Matched edges from the odd-node pairing step.

    Returns
    -------
    mst_added : ndarray, shape (M, 2) or None
        New cross-component edges added by the minimum spanning tree, or
        ``None`` if the graph was already connected and no reconnection was
        needed.
    """
    n = A.shape[0]
    orig = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] > 0]
    Gc = nx.Graph()
    Gc.add_edges_from(orig)
    for (u, v) in np.atleast_2d(added):
        if np.size(added):
            Gc.add_edge(int(u), int(v))
    if Gc.number_of_nodes() == 0 or nx.number_connected_components(Gc) <= 1:
        return None
    edge_nodes = sorted({x for e in orig for x in e})
    existing = {tuple(sorted(e)) for e in orig}
    # _RECONNECT_ODD_ONLY restricts reconnect candidates to odd-degree nodes
    # (crack endpoints/junctions) to avoid mid-crack pass-through landings.
    # Falls back to all edge-nodes when odd-only cannot span every component.
    cand = edge_nodes
    if _RECONNECT_ODD_ONLY:
        deg = {nn: 0 for nn in edge_nodes}
        for (u, v) in orig:
            deg[u] += 1; deg[v] += 1
        odd = [nn for nn in edge_nodes if deg[nn] % 2 == 1]
        Gtest = nx.Graph(); Gtest.add_nodes_from(edge_nodes)
        Gtest.add_edges_from(orig)
        Gtest.add_edges_from((u, v) for ai, u in enumerate(odd) for v in odd[ai + 1:])
        if nx.number_connected_components(Gtest) <= 1:
            cand = odd
    Gg = nx.Graph()
    Gg.add_nodes_from(edge_nodes)
    for (u, v) in orig:
        Gg.add_edge(u, v, weight=0.0)
    for ai, u in enumerate(cand):
        for v in cand[ai + 1:]:
            if tuple(sorted((u, v))) not in existing:
                Gg.add_edge(u, v, weight=float(DIST[u, v]))
    T = nx.minimum_spanning_tree(Gg, algorithm="kruskal")
    mst_added = np.array([[u, v] for (u, v) in T.edges()
                          if tuple(sorted((u, v))) not in existing], dtype=int)
    for (u, v) in mst_added:
        if A[u, v] == 0:
            A[u, v] = DIST[u, v]
            A[v, u] = DIST[u, v]
        ADJ[u, v] += 1
        ADJ[v, u] += 1
    return mst_added


def ChinesePostman(ADJ, Matrix_Input, DIST, Start=None, ee=None, cl=None):
    """Find a minimum-cost closed traversal that covers every required graph edge.

    Solves the Chinese Postman Problem on the graph described by ``Matrix_Input``
    (edge weights) and ``ADJ`` (edge multiplicities).  If the graph already has
    an Euler circuit (all node degrees even), Fleury's algorithm is applied
    directly.  Otherwise, the minimum-weight set of extra edges is found via a
    binary Integer Linear Program (ILP) to make all degrees even; the graph is
    then augmented and Fleury's algorithm is applied.

    When ``Start`` or ``cl`` is provided, the traversal begins at that node and
    the matched start edge is removed so that Fleury produces an open Euler path
    (useful for chaining across multiple planning iterations).

    Parameters
    ----------
    ADJ : array-like, shape (N, N)
        Edge-count (multiplicity) adjacency matrix.  Parallel edges are encoded
        as integer counts > 1.
    Matrix_Input : array-like, shape (N, N)
        Weighted adjacency matrix.  ``Matrix_Input[i, j]`` is the weight (e.g.
        travel cost) of edge ``(i, j)``; 0 means no edge.
    DIST : array-like, shape (N, N)
        All-pairs shortest-path distance matrix.  Used by the odd-node ILP to
        find minimum-cost augmenting edges.
    Start : int or None, optional
        Preferred start node (0-based).  Ignored when ``cl`` is provided.
    ee : array-like or None, optional
        Edges to exclude from the odd-node pairing candidates.
    cl : int or None, optional
        Forced start node (0-based).  Takes precedence over ``Start``.

    Returns
    -------
    Path : list of int
        Ordered node sequence of the traversal (0-based node indices).
    Weight : float
        Total traversal cost (sum of all traversed edge weights, including
        augmented edges).
    add : ndarray, shape (K, 2)
        The augmenting edges added to make the graph Eulerian; empty if the
        input was already Eulerian or after reconnection.
    st : ndarray, shape (1, 2) or (0, 2)
        The removed start edge ``[u, v]``, if any, used to open the circuit into
        a path from ``cl`` / ``Start``; empty otherwise.
    """
    A = np.asarray(Matrix_Input, dtype=float).copy()
    ADJ = np.asarray(ADJ, dtype=float).copy()
    DIST = np.asarray(DIST, dtype=float)
    t = _euler_test(A)

    if t == 1:                               # Eulerian: apply Fleury directly
        start = int(cl) if cl is not None and np.size(cl) else (int(Start) if Start is not None and np.size(Start) else 0)
        Path, Weight = _fleury(A, ADJ, start)
        return Path, Weight, np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    # Non-Eulerian (semi-Eulerian t==2 also handled here): match odd nodes,
    # augment the graph, pick the start node, and — when cl/Start is given —
    # remove the matched start edge so Fleury produces an open Euler path.
    added, added_w, st = _opt_odd_pair(ADJ, DIST, ee, cl, False, True)
    if added.size == 0:
        added, added_w, st = _opt_odd_pair(ADJ, DIST, ee, cl, False, False)
    add = added.copy()

    # tempG: original graph before augmentation, used to distinguish original vs matched edges
    tempG = nx.Graph()
    tempG.add_nodes_from(range(A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if A[i, j] > 0:
                tempG.add_edge(i, j)

    # If the matched graph is still disconnected, MST-reconnect and re-match
    mst_added = _circuit_reconnect(A, ADJ, DIST, added)
    if mst_added is not None:
        add = mst_added.copy() if mst_added.size else np.empty((0, 2), dtype=int)
        added, added_w, st = _opt_odd_pair(ADJ, DIST, ee, cl, False, True)
        if added.size:
            add = np.vstack([add, added]) if add.size else added.copy()

    for (u, v) in added:
        if A[u, v] == 0:
            A[u, v] = DIST[u, v]
            A[v, u] = DIST[u, v]
        ADJ[u, v] += 1
        ADJ[v, u] += 1

    # When the sole matched edge equals the start edge, reset st to the odd nodes
    if add.shape[0] == 1 and st.size and np.array_equal(add[0], st[0]):
        deg = (_bin(A).sum(axis=1) % 2 != 0)
        st = np.flatnonzero(deg).reshape(1, -1)
        if st.size == 0:
            st = np.empty((0, 2), dtype=int)

    def _remove_start_edge(start):
        """Pick max-weight non-original neighbour edge at `start`, remove it, return st."""
        nbrs = sorted(int(c) for c in np.flatnonzero(_bin(A)[start]))
        if not nbrs:
            return np.empty((0, 2), dtype=int)
        orig = np.array([tempG.has_edge(start, nb) for nb in nbrs])
        if orig.sum() == len(nbrs):
            return st_local
        cand = [nb for nb, o in zip(nbrs, orig) if not o]
        wmax = max(A[start, nb] for nb in cand)
        pick = next(nb for nb in nbrs if A[start, nb] == wmax)
        A[start, pick] = 0.0
        A[pick, start] = 0.0
        ADJ[start, pick] -= 1
        ADJ[pick, start] -= 1
        return np.array([[start, pick]], dtype=int)

    st_local = st
    if cl is not None and np.size(cl):
        start = int(cl)
        st = _remove_start_edge(start)
    elif Start is not None and np.size(Start):
        start = int(Start)
        st = _remove_start_edge(start)
    elif st.size:
        start = int(st[0, 0])
    else:
        start = 0

    # Matched edges live in ADJ (incremented counts); pass an empty P so Fleury
    # traverses them via the max-count edge ordering rather than the matched-edge list.
    Path, Weight = _fleury(A, ADJ, start, P=np.empty((0, 2)), ow=added_w)

    # If the path opens along the removed start edge, reverse it
    if st.size and st.shape[1] == 2 and len(Path) >= 2 and \
            Path[0] == int(st[0, 0]) and Path[1] == int(st[0, 1]):
        Path = Path[::-1]
    return Path, Weight, add, st

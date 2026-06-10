"""Geometry-routed coverage-and-fill path via a cell Rural-Postman Problem formulation.

This module computes the robot's full coverage-and-crack-filling path for the
offline Sensor-based Complete Coverage (SCC) planner using a deterministic
Rural-Postman Problem (RPP) formulation that operates directly on the coverage
cell geometry.

**How it works**

Each coverage cell contributes one required sweep edge to an undirected
multigraph: the edge connects the cell's two boustrophedon entry/exit corner
ports and its weight equals the estimated sweep length (cell area divided by
the sweep-line spacing).  The crack graph contributes additional required edges
connecting crack endpoints and branch points.  The RPP is then solved in three
steps:

1. **Min-weight T-join** — odd-degree nodes are paired by minimum-weight
   matching to make every node even-degree (Euler condition).  Pairs are chosen
   from geometric straight-line distances with a retrace penalty; sweep edges
   are excluded from matching to avoid reversing a cell mid-tour.

2. **Component reconnection** — if the required-edge graph has multiple
   connected components, they are joined by minimum-spanning-tree connectors
   (each connector is doubled to preserve parity).

3. **Euler circuit** — a closed Euler circuit is extracted from the resulting
   even-degree graph.  All node identities are first replaced by canonical
   geometric ranks (sorted by ``(x, y)``) so the result is independent of the
   order in which cells and crack edges were supplied.

The circuit is then assembled into a continuous ``(x, y)`` path by choosing
the boustrophedon sweep orientation for each cell that best aligns with the
Euler circuit's traversal direction, using a dynamic-programming pass that
minimises total transit (connection) distance.  A wall-clamping pass at the
end pulls path points away from workspace boundaries to at least one
sweep-spacing unit, except near cracks that require the robot to go closer.

**Entry point**

``route(node, crackEdge, cell_geoms, cell_verts, s, a)``
    Returns the full ``(N, 2)`` ``(x, y)`` coverage-and-fill path.
"""
import math

import numpy as np
import networkx as nx

import private.poly_utils as pu
from BoustrophedonPath import BoustrophedonPath

SPACING = 342.0 / math.sqrt(2)     # s/sqrt(2) sweep-line spacing


def _loops(V):
    """Largest (outer) loop of a NaN-separated (x,y) vertex array."""
    V = np.atleast_2d(V)
    out, cur = [], []
    for row in V:
        if np.isnan(row[0]):
            if cur:
                out.append(np.array(cur)); cur = []
        else:
            cur.append(row)
    if cur:
        out.append(np.array(cur))
    return max(out, key=_area) if out else np.empty((0, 2))


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _plen(poly):
    p = np.atleast_2d(poly)
    return float(np.hypot(*(np.diff(p, axis=0).T)).sum()) if len(p) > 1 else 0.0


def cell_options(geom, s, a):
    """Enumerate the boustrophedon sweep orientations available for a single cell.

    For each combination of starting side (left or right boundary of the cell)
    and zig-zag direction (two choices), the boustrophedon path through the
    cell is computed.  Each orientation yields a different pair of entry and
    exit corner points and a different polyline length.

    Parameters
    ----------
    geom : shapely geometry (Polygon or MultiPolygon)
        Cell boundary polygon.
    s : float
        Sensor diameter in pixels.  The sweep-line spacing is ``s / sqrt(2)``.
    a : float
        Robot nozzle reach (footprint radius) in pixels.

    Returns
    -------
    list of (entry, exit, polyline, length) tuples
        ``entry`` and ``exit`` are ``(x, y)`` float arrays giving the inset
        sweep corner points.  ``polyline`` is an ``(M, 2)`` float array of the
        full boustrophedon path.  ``length`` is the polyline arc length in
        pixels.  Up to four orientations are returned; at least one is always
        present.
    """
    cell = pu.PolyShape(geom)
    loop = _loops(np.atleast_2d(np.asarray(geom.exterior.coords)) if geom.geom_type == "Polygon"
                  else np.vstack([np.asarray(g.exterior.coords) for g in geom.geoms]))
    pL = loop[int(np.argmin(loop[:, 0]))]; pR = loop[int(np.argmax(loop[:, 0]))]
    bp = s / math.sqrt(2); opts = []
    for start_left in (True, False):
        a0, a1 = (pL, pR) if start_left else (pR, pL)
        allNode = np.array([[a0[1], a0[0]], [a1[1], a1[0]]], float)   # (row,col)=(y,x); Start=row0
        for d in (0, 1):
            subXY, _ = BoustrophedonPath(cell, cell, np.array([0, 1]), bp, d, 0, 0,
                                         True, False, allNode, s, a)
            subXY = np.atleast_2d(np.asarray(subXY, float))
            if subXY.size:
                opts.append((subXY[0].copy(), subXY[-1].copy(), subXY, _plen(subXY)))
    return opts or [(pL, pR, np.array([pL, pR], float), _plen(np.array([pL, pR])))]


def _corner_ports(geom, s, a):
    """Return the two inset sweep corner ports for a cell (canonical orientation).

    The canonical orientation is start-left, direction 0 — the lexicographically
    smallest entry point across all four boustrophedon variants.  Using real
    inset corners (rather than x-extreme boundary points) as RPP ports ensures
    that the matching and Euler-circuit edge order reflects the actual sweep
    geometry.
    """
    opts = cell_options(geom, s, a)
    e0, x0, _p, _L = min(opts, key=lambda q: (round(float(q[0][0]), 1), round(float(q[0][1]), 1)))
    return np.asarray(e0, float), np.asarray(x0, float)


def _build_graph(node, crackEdge, cell_verts, cell_geoms, s, a, corner_ports):
    """Build the required-edge graph for the Rural-Postman Problem.

    The graph contains one required edge per crack segment (connecting its
    endpoint nodes) and one required sweep edge per coverage cell (connecting
    the cell's two port points).  Crack edges use the straight-line distance
    between crack nodes as their weight; sweep edges use the cell area divided
    by the sweep-line spacing as a proxy for sweep length.

    When ``corner_ports=True`` the cell ports are the inset boustrophedon
    corner points; when ``False`` they are the x-extreme boundary points.
    """
    nfx = np.atleast_2d(node)[:, ::-1].astype(float)        # -> (x,y)
    coords = [tuple(p) for p in nfx]
    req = [(int(u), int(v), float(np.hypot(*(nfx[u] - nfx[v]))))
           for u, v in np.atleast_2d(np.asarray(crackEdge, int))]
    sweep_edges, cell_of = set(), {}
    for c in range(len(cell_geoms)):
        if corner_ports:
            pl, pr = _corner_ports(cell_geoms[c], s, a)
        else:
            loop = _loops(cell_verts[c])
            pl = loop[int(np.argmin(loop[:, 0]))]; pr = loop[int(np.argmax(loop[:, 0]))]
        w = _area(_loops(cell_verts[c])) / SPACING
        iL = len(coords); coords.append(tuple(pl))
        iR = len(coords); coords.append(tuple(pr))
        req.append((iL, iR, float(w)))
        sweep_edges.add((iL, iR)); cell_of[(iL, iR)] = c
    return np.array(coords, float), req, sweep_edges, cell_of


def _canonical_ids(coords):
    """Assign canonical integer node IDs sorted by ``(x, y)`` coordinate.

    Relabelling before the Euler circuit ensures the result is independent of
    the order in which cells and crack edges were supplied.
    """
    order = sorted(range(len(coords)), key=lambda i: (round(coords[i][0], 3), round(coords[i][1], 3)))
    cid = np.empty(len(coords), int)
    for rank, i in enumerate(order):
        cid[i] = rank
    return cid


def _solve_rpp(coords, req, sweep_edges, cell_of):
    """Solve the Rural-Postman Problem and return a typed Euler circuit.

    Steps: (1) pair odd-degree nodes by min-weight matching to make the graph
    Eulerian; (2) join disconnected components with minimum-spanning-tree
    connectors (each doubled to preserve parity); (3) extract an Euler circuit.
    All node ids are first replaced by canonical geometric ranks for
    deterministic output.

    Returns a list of ``(type, cell, p0, p1)`` tuples where ``type`` is
    ``"sweep"``, ``"crack"``, or ``"transit"``; ``cell`` is the cell index for
    sweep edges and ``None`` otherwise; and ``p0``/``p1`` are ``(x, y)`` float
    arrays.
    """
    cid = _canonical_ids(coords)
    inv = {int(cid[i]): i for i in range(len(coords))}
    def C(i): return int(cid[i])

    G = nx.MultiGraph(); G.add_nodes_from(range(len(coords)))
    crack_deg = {}
    for (u, v, w) in req:
        cell = cell_of.get((u, v), cell_of.get((v, u)))
        G.add_edge(C(u), C(v), weight=float(w), required=True, cell=cell)
        if cell is None:                                  # crack edge -> tally crack degree
            crack_deg[C(u)] = crack_deg.get(C(u), 0) + 1
            crack_deg[C(v)] = crack_deg.get(C(v), 0) + 1
    # mid-crack nodes (even crack degree): connectors must NOT attach here (the route would
    # arrive in the middle of a crack). Both the T-join (matches odd nodes only) and the MST
    # component reconnection avoid them.
    avoid = {n for n, d in crack_deg.items() if d % 2 == 0 and d > 0}

    def dist(a, b):
        pa, pb = coords[inv[a]], coords[inv[b]]
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    forbid = {(min(C(u), C(v)), max(C(u), C(v))) for (u, v) in sweep_edges}

    odd = sorted(v for v in G.nodes if G.degree(v) % 2 == 1)
    if odd:
        gdist = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        H = nx.Graph(); big = max(dist(x, y) for x in odd for y in odd) + 1.0
        for ii in range(len(odd)):
            for jj in range(ii + 1, len(odd)):
                x, y = odd[ii], odd[jj]
                if (min(x, y), max(x, y)) in forbid:
                    continue
                e = dist(x, y); gd = gdist.get(x, {}).get(y, float("inf"))
                penalty = big if (gd < float("inf") and e >= 0.85 * gd) else 0.0   # retrace penalty
                H.add_edge(x, y, weight=big - e - penalty)
        for a, b in sorted((min(x, y), max(x, y)) for x, y in nx.max_weight_matching(H, maxcardinality=True)):
            G.add_edge(a, b, weight=dist(a, b), required=False, cell=None)

    comps = [sorted(c) for c in nx.connected_components(G)]
    if len(comps) > 1:
        Gc = nx.Graph(); pair = {}
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                ci = [n for n in comps[i] if n not in avoid] or comps[i]    # avoid mid-crack attach
                cj = [n for n in comps[j] if n not in avoid] or comps[j]
                best = min((dist(a, b), a, b) for a in ci for b in cj)
                Gc.add_edge(i, j, weight=best[0]); pair[(i, j)] = (best[1], best[2])
        for i, j in nx.minimum_spanning_edges(Gc, data=False):
            a, b = pair[(min(i, j), max(i, j))]
            G.add_edge(a, b, weight=dist(a, b), required=False, cell=None)
            G.add_edge(a, b, weight=dist(a, b), required=False, cell=None)   # doubled -> parity

    euler = []
    for u, v, k in nx.eulerian_circuit(G, source=min(G.nodes), keys=True):
        e = G.edges[u, v, k]
        p0 = np.array(coords[inv[u]]); p1 = np.array(coords[inv[v]])
        cell = e.get("cell")
        typ = "sweep" if cell is not None else ("crack" if e.get("required") else "transit")
        euler.append((typ, cell, p0, p1))
    return euler


def _assemble(euler, cell_geoms, s, a, return_transit=False):
    """Assemble the Euler circuit into a continuous path using dynamic programming.

    For each required element in the circuit (sweep or crack), all orientation
    variants (entry/exit combinations) are considered.  A DP pass over the
    ordered elements chooses the orientation sequence that minimises total
    straight-line transit distance between consecutive elements.

    Parameters
    ----------
    euler : list of (type, cell, p0, p1) tuples
        Typed Euler circuit from ``_solve_rpp``.
    cell_geoms : list of shapely geometries
        Cell boundary polygons (indexed by the ``cell`` field in ``euler``).
    s, a : float
        Sensor diameter and nozzle reach in pixels.
    return_transit : bool, optional
        If ``True``, also return the list of transit-connector endpoint pairs.

    Returns
    -------
    PE : numpy.ndarray, shape (N, 2), float
        Assembled ``(x, y)`` path.
    transit : list of (p0, p1) pairs (only when ``return_transit=True``)
        Endpoints of the straight-line transit connectors inserted between
        consecutive required elements.
    """
    elems = []
    for typ, cell, p0, p1 in euler:
        if typ == "crack":
            L = float(np.hypot(*(p1 - p0)))
            elems.append([("crk", p0, p1, np.array([p0, p1]), L),
                          ("crk", p1, p0, np.array([p1, p0]), L)])
        elif typ == "sweep":
            elems.append([("cell", o[0], o[1], o[2], o[3]) for o in cell_options(cell_geoms[cell], s, a)])
            elems[-1] = list(elems[-1])
    if not elems:
        return (np.empty((0, 2)), []) if return_transit else np.empty((0, 2))
    N = len(elems); INF = float("inf")
    dp = [(o[4], -1) for o in elems[0]]; back = [[-1] * len(elems[0])]
    for i in range(1, N):
        ndp, bi = [], []
        for o in elems[i]:
            best, bj = INF, -1
            for j, pj in enumerate(elems[i - 1]):
                c = dp[j][0] + float(np.hypot(*(o[1] - pj[2]))) + o[4]
                if c < best:
                    best, bj = c, j
            ndp.append((best, bj)); bi.append(bj)
        dp = ndp; back.append(bi)
    chosen = [0] * N; chosen[N - 1] = int(np.argmin([c for c, _ in dp]))
    for i in range(N - 1, 0, -1):
        chosen[i - 1] = back[i][chosen[i]]
    segs = []; prev_exit = None; transit = []
    for i in range(N):
        o = elems[i][chosen[i]]
        if prev_exit is not None:
            segs.append(np.array([prev_exit, o[1]]))
            transit.append((np.asarray(prev_exit, float), np.asarray(o[1], float)))
        segs.append(o[3]); prev_exit = o[2]
    PE = np.vstack(segs)
    return (PE, transit) if return_transit else PE


def _assemble_euler(euler, cell_geoms, s, a):
    """Assemble the Euler circuit into a path preserving the circuit's transit edges.

    Unlike the DP assembler, transit connectors follow the Euler circuit
    exactly (touching only odd-degree nodes — crack tips, branch points, and
    cell corner ports) rather than introducing new straight-line connections.
    This prevents connectors from landing on even-degree mid-crack nodes.
    Each cell is swept in the orientation whose entry/exit points best match
    the two Euler-circuit endpoints of its sweep edge; small gaps between
    corner and port are bridged with short straight segments.

    Parameters
    ----------
    euler : list of (type, cell, p0, p1) tuples
        Typed Euler circuit from ``_solve_rpp``.
    cell_geoms : list of shapely geometries
        Cell boundary polygons.
    s, a : float
        Sensor diameter and nozzle reach in pixels.

    Returns
    -------
    numpy.ndarray, shape (N, 2), float
        Assembled ``(x, y)`` path.
    """
    cache, segs = {}, []
    for typ, cell, p0, p1 in euler:
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        if typ == "sweep":
            if cell not in cache:
                cache[cell] = cell_options(cell_geoms[cell], s, a)
            opts = cache[cell]
            fwd = min(opts, key=lambda q: np.hypot(*(q[0] - p0)) + np.hypot(*(q[1] - p1)))
            rev = min(opts, key=lambda q: np.hypot(*(q[0] - p1)) + np.hypot(*(q[1] - p0)))
            df = np.hypot(*(fwd[0] - p0)) + np.hypot(*(fwd[1] - p1))
            dr = np.hypot(*(rev[0] - p1)) + np.hypot(*(rev[1] - p0))
            segs.append(np.atleast_2d(np.asarray(rev[2][::-1] if dr < df else fwd[2], float)))
        else:                                          # crack / transit: node-to-node (odd-only)
            segs.append(np.array([p0, p1]))
    PE, prev = [], None
    for poly in segs:
        if prev is not None and np.hypot(*(poly[0] - prev)) > 1e-6:
            PE.append(np.array([prev, poly[0]]))       # bridge tiny corner<->port gaps
        PE.append(poly); prev = poly[-1]
    return np.vstack(PE) if PE else np.empty((0, 2))


def _wall_clamp(PE, node, crackEdge, s, W=3048.0, H=2896.0):
    """Pull path points away from workspace walls to at least one sweep-spacing unit.

    Each path point is clamped to be at least ``s / sqrt(2)`` pixels from each
    of the four workspace boundaries, unless a crack node lies within that
    distance of the same wall — in which case the robot must approach closer to
    fill the crack and the point is left unchanged.

    Parameters
    ----------
    PE : numpy.ndarray, shape (N, 2), float
        Path waypoints in ``(x, y)`` pixel coordinates.
    node : array-like, shape (V, 2)
        Crack node coordinates in ``(row, col)`` order.
    crackEdge : array-like, shape (E, 2), int
        Pairs of crack node indices forming crack edges.
    s : float
        Sensor diameter in pixels.
    W, H : float
        Workspace width and height in pixels.

    Returns
    -------
    numpy.ndarray, shape (N, 2), float
        Clamped path waypoints.
    """
    SP = s / math.sqrt(2)
    nfx = np.atleast_2d(node)[:, ::-1].astype(float)            # crack nodes (x,y)
    pts = [nfx]                                                 # + densely sampled crack edges
    for u, v in np.atleast_2d(np.asarray(crackEdge, int)):
        A, B = nfx[u], nfx[v]
        pts.append(np.linspace(A, B, max(2, int(np.hypot(*(B - A)) / 40))))
    cp = np.vstack(pts)
    out = np.array(PE, float)
    cx, cy = cp[:, 0], cp[:, 1]
    for i in range(out.shape[0]):
        x, y = out[i]
        if y < SP and not (((cy < SP) & (np.abs(cx - x) < SP)).any()):       out[i, 1] = SP
        if y > H - SP and not (((cy > H - SP) & (np.abs(cx - x) < SP)).any()): out[i, 1] = H - SP
        if x < SP and not (((cx < SP) & (np.abs(cy - y) < SP)).any()):       out[i, 0] = SP
        if x > W - SP and not (((cx > W - SP) & (np.abs(cy - y) < SP)).any()): out[i, 0] = W - SP
    return out


def _even_crack_nodes(node, crackEdge):
    """Return the ``(x, y)`` coordinates of even-degree crack nodes.

    Even-degree crack nodes (degree > 0) are mid-crack continuation points
    where the crack passes through rather than terminates.  A transit connector
    that ends at one of these points would interrupt the crack traversal.
    """
    ce = np.atleast_2d(np.asarray(crackEdge, int))
    deg = np.zeros(np.atleast_2d(node).shape[0], int)
    for u, v in ce:
        deg[u] += 1; deg[v] += 1
    nfx = np.atleast_2d(node)[:, ::-1].astype(float)         # (x,y)
    return nfx[(deg % 2 == 0) & (deg > 0)]


def _has_midcrack(transit, even_xy, tol=6.0, min_len=15.0):
    """Return True if any non-trivial transit connector endpoint lands on a mid-crack node.

    A connector shorter than ``min_len`` pixels is treated as a crack
    continuation (two edges sharing a node) rather than a genuine transit move,
    and is excluded from the check to avoid false positives.

    Parameters
    ----------
    transit : list of (p0, p1) pairs
        Transit connector endpoint coordinates ``(x, y)``.
    even_xy : numpy.ndarray, shape (M, 2), float
        Coordinates of even-degree crack nodes.
    tol : float
        Distance tolerance in pixels for a point to be considered "on" a node.
    min_len : float
        Connectors shorter than this are ignored.

    Returns
    -------
    bool
        ``True`` if a real transit connector lands within ``tol`` pixels of any
        even-degree crack node.
    """
    if len(even_xy) == 0 or not transit:
        return False
    for p0, p1 in transit:
        if np.hypot(*(p1 - p0)) < min_len:               # continuation / zero-length -> skip
            continue
        for q in (p0, p1):
            if (np.hypot(even_xy[:, 0] - q[0], even_xy[:, 1] - q[1]) <= tol).any():
                return True
    return False


def route(node, crackEdge, cell_geoms, cell_verts, s, a):
    """Compute the full coverage-and-fill path using the cell Rural-Postman formulation.

    Tries two port-selection strategies (x-extreme boundary points and inset
    boustrophedon corner points) and, for each, selects the assembler that
    produces a clean path: the DP assembler is used when its transit connectors
    do not land on mid-crack nodes; otherwise the Euler-walk assembler (which
    inherits the circuit's odd-only connectors) is used instead.  The shorter
    of the two resulting paths is returned after wall clamping.

    Parameters
    ----------
    node : array-like, shape (V, 2)
        Crack graph node coordinates in ``(row, col)`` order.
    crackEdge : array-like, shape (E, 2), int
        Pairs of node indices forming the required crack edges.
    cell_geoms : list of shapely geometries
        Coverage cell boundary polygons (one per cell).
    cell_verts : list of array-like
        NaN-separated ``(x, y)`` vertex arrays for each cell (used for
        x-extreme port computation).
    s : float
        Sensor diameter in pixels.
    a : float
        Nozzle reach (footprint radius) in pixels.

    Returns
    -------
    numpy.ndarray, shape (N, 2), float
        The coverage-and-fill path as ``(x, y)`` waypoints.  Returns an empty
        array if the graph cannot be solved.
    """
    even_xy = _even_crack_nodes(node, crackEdge)
    best = None
    for corner_ports in (False, True):
        try:
            coords, req, se, cof = _build_graph(node, crackEdge, cell_verts, cell_geoms, s, a, corner_ports)
            euler = _solve_rpp(coords, req, se, cof)
        except Exception:
            continue
        try:
            PE, transit = _assemble(euler, cell_geoms, s, a, return_transit=True)
            if _has_midcrack(transit, even_xy):              # DP lands mid-crack -> odd-only
                PE = _assemble_euler(euler, cell_geoms, s, a)
            if PE.shape[0] >= 2:
                PE = _wall_clamp(PE, node, crackEdge, s)
                L = _plen(PE)
                if best is None or L < best[0]:
                    best = (L, PE)
        except Exception:
            continue
    return best[1] if best is not None else np.empty((0, 2))

"""Offline (known-map) crack-graph extraction for the SCC planner.

Given a crack map (PNG name, or a pre-computed skeleton array), skeletonizes the
crack binary image, traces crack polylines, connects them into a crack graph, finds
sensor-footprint overlap nodes, and builds a visibility graph.

Returns ``(node, edgeList, ttt)``:

- ``node``     : ``(N, 2)`` crack-graph node coordinates in **image (row, col)** order.
- ``edgeList`` : ``(E, 2)`` **0-based** undirected node-index pairs.
- ``ttt``      : timing placeholder (0.0; unused).

Two crack-graph formulations are provided:

1. **Active path** (``ImagePlanning_SCC``): taut-string centerline graph — every crack
   centerline is approximated with a minimum-length polyline whose straight chords stay
   within the footprint radius, guaranteeing 100 % footprint coverage at minimum cost.

2. **Legacy path** (``_ImagePlanning_SCC_chord_legacy``): chord-based endpoint/visibility
   graph — kept for reference and regression testing.

Crack data is stored in ``(row, col)`` order throughout; polygon buffers are constructed
in ``(col, row)`` order and converted back for interior / visibility queries.
ChinesePostman routing is performed in ``SCC.py``; this module stops at ``edgeList``.
"""

from math import sqrt

import numpy as np

from private import poly_utils as pu
from private.utils import inpxMap, spdist, total_length
from private.compCrack import compCrack_branching as compCrack
from private.DecimatePoly import DecimatePoly
from private.line_of_sight import line_of_sight_poly
from private.pathfinder import pathfinder
from private.Polygons_Intersection import Polygons_Intersection
from ImagePlanning_oSCC import (
    ab2v, _dsearchn, _ismember_rows, _clip_outside, _intersect_nonempty)

_DIR_MAP = np.array([[-1, -1], [-1, 0], [-1, 1],
                     [0, -1], [0, 1],
                     [1, -1], [1, 0], [1, 1]])


def _front_end(img_n, a, skel=None):
    if skel is not None:
        BW3 = (np.asarray(skel) > 0).astype(int)
    elif isinstance(img_n, str) and "Gauss" in img_n:
        # Gaussian map: load the pre-computed crack accumulator from a .mat file.
        # Accepts a path of the form 'Gaussian<b>/<name>' or a bare name (searched across Gaussian1..8).
        import os
        from scipy.io import loadmat
        from private.bwskel import bwskel
        from OnlineSCC import _bwmorph_fill
        if os.sep in img_n or "/" in img_n:
            matpath = os.path.join("CrackMaps", img_n.replace("/", os.sep) + ".mat")
        else:
            matpath = next((p for b in range(1, 9)
                            for p in [os.path.join("CrackMaps", f"Gaussian{b}", img_n + ".mat")]
                            if os.path.exists(p)), None)
            if matpath is None:
                raise FileNotFoundError(f"Gaussian map '{img_n}' not found in CrackMaps/Gaussian1..8")
        BW = np.asarray(loadmat(matpath)["crackGen"]) > 0
        BW = BW.copy(); BW[-1, -1] = False                          # clear sentinel pixel
        BW2 = _bwmorph_fill(BW)
        BW2 = bwskel(BW2)
        BW3 = _prune_branches(BW2, a).astype(int)
    else:
        from skimage.io import imread
        from skimage.filters import threshold_otsu
        from skimage.morphology import remove_small_objects
        from private.bwskel import bwskel
        from OnlineSCC import _bwmorph_fill

        img = imread(f"CrackMaps/Uniform/{img_n}.png")
        ch = img[:, :, 0] if img.ndim == 3 else img
        BW = ch > threshold_otsu(ch)
        BW = remove_small_objects(BW, max_size=51, connectivity=2)
        BW = ~BW
        BW2 = _bwmorph_fill(BW)
        BW2 = bwskel(BW2)
        BW3 = _prune_branches(BW2, a)                                # remove short side-branches (length <= a)
        BW3 = BW3.astype(int)
    return BW3


# 8-connectivity neighbor offsets (row delta, col delta)
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _prune_branches(BW, a):
    """Remove side-branches whose arc-length (endpoint to branch point) is <= ``a``.

    Short outlier spurs that stick off a larger crack are covered by the robot footprint
    while traversing the main crack, so planning separate fill paths for them is wasted
    motion. Unlike iterative spur erosion, this removes an entire branch only when it
    terminates at a branch point (not at another endpoint), so true crack tips are never
    shortened.
    """
    S = set(map(tuple, np.argwhere(np.asarray(BW) > 0).tolist()))
    while True:
        deg = {p: sum(((p[0] + dr, p[1] + dc) in S) for dr, dc in _NB8) for p in S}
        to_remove = set()
        for ep in [p for p, d in deg.items() if d == 1]:
            path = [ep]; prev = None; cur = ep
            while True:                                  # walk the spur until a junction
                nb = [(cur[0] + dr, cur[1] + dc) for dr, dc in _NB8
                      if (cur[0] + dr, cur[1] + dc) in S and (cur[0] + dr, cur[1] + dc) != prev]
                if deg[cur] != 2 and cur != ep:
                    break
                if not nb:
                    break
                prev, cur = cur, nb[0]
                path.append(cur)
                if deg[cur] != 2:
                    break
            term = path[-1]
            if deg[term] >= 3:                           # ends at a branch point -> side-branch
                seg = np.asarray(path, float)
                length = float(np.hypot(*np.diff(seg, axis=0).T).sum()) if len(seg) > 1 else 0.0
                if length <= a:
                    to_remove.update(path[:-1])          # keep the branch point itself
        if not to_remove:
            break
        S -= to_remove
    out = np.zeros(np.asarray(BW).shape, dtype=int)
    if S:
        idx = np.array(list(S)); out[idx[:, 0], idx[:, 1]] = 1
    return out


def _acute_fP(crackRaw, fP, a, yy, mark_det):
    """Compute the acute-angle adjusted footprint offset for each crack segment.

    For each crack whose end-to-end vector makes an acute angle with the reference
    vertical, adjusts the footprint offset ``fP`` to account for the angular geometry.

    Parameters
    ----------
    crackRaw : list of array
        Crack polyline coordinates ``(row, col)``.
    fP : ndarray
        Per-crack footprint offset values; updated in place.
    a : float
        Robot footprint radius in pixels.
    yy : int
        Image width (number of columns).
    mark_det : bool
        If True, also collect and return the indices of acute cracks (``det``).
        If False, only update ``fP``.

    Returns
    -------
    fP : ndarray
    det : list of int
        Indices of acute cracks (empty when ``mark_det=False``).
    """
    det = []
    for l, cr in enumerate(crackRaw):
        cr = np.asarray(cr, dtype=float)
        if total_length(cr) > 2 * a * sqrt(2):
            aRan = pu.polybuffer(cr[[0, -1]], a, kind="points")
            inside = pu.isinterior(aRan, cr)
            crackW = cr[~np.asarray(inside, bool)]
            mov = (crackW - crackW[0]) if crackW.shape[0] else (cr - cr[0])
        else:
            mov = cr - cr[0]
        v1 = np.array([0.0, yy])
        v2 = mov[-1]
        ang = ab2v(v1, v2)
        ar = np.radians(ang)
        rot = np.array([[np.cos(ar), np.sin(ar)], [-np.sin(ar), np.cos(ar)]])
        rv2 = (rot @ mov.T).T
        cand = [int(np.argmin(rv2[:, 0])), int(np.argmax(rv2[:, 0]))]
        peak = cand[int(np.argmax(np.abs(rv2[cand, 0])))]
        v1 = -rv2[peak]
        v2 = rv2[-1] - rv2[peak]
        ang = ab2v(v1, v2) / 2
        if ang < 45 or ang > 360 - 45:
            fP[l] = np.tan(np.radians(ang)) * a
            if mark_det:
                det.append(l)
    return fP, det


def _acute(ang):
    return ang < 45 or ang > 360 - 45


def _connect_graph_offline(line, crackRaw, link, pointX, pointY, fP, a):
    """Single-pass crack-graph connection for the offline (known-map) case.

    Iterates through crack segments and greedily chains them: if the tail of the
    current chain is within ``a`` pixels of another segment's start or end, and the
    junction angle is not acute, the segments are merged into one polyline.  Segments
    that were merged into another are removed from the output.

    Returns updated ``(line, crackRaw, link, fP)`` with merged segments removed.
    """
    n = line.shape[0]
    if n == 0:
        return line, crackRaw, link, fP
    crackRawt = [np.asarray(c, float).copy() for c in crackRaw]
    detind = []
    templine = line.copy().astype(float)

    intt = 0
    link[intt]["x"] = pointX[intt].copy()
    link[intt]["y"] = pointY[intt].copy()
    templine[intt] = 0.0
    intt2 = list(range(1, n))
    indt = 0

    while intt2:
        tail = np.array([link[intt]["x"][-1], link[intt]["y"][-1]], float)
        tempdist = spdist(tail, templine[:, 0:2])           # distance to segment starts
        inst_set = np.flatnonzero(tempdist <= a)
        ang = ang2 = None
        if inst_set.size:
            mn = tempdist[inst_set].min()
            inst = int(np.flatnonzero(tempdist == mn)[0])
            v1 = crackRawt[indt][0] - crackRawt[indt][-1]
            v2 = np.asarray(crackRaw[intt], float)[0] - np.asarray(crackRaw[intt], float)[-1]
            v3 = np.asarray(crackRaw[inst], float)[-1] - np.asarray(crackRaw[inst], float)[0]
            ang = ab2v(v1, v3)
            indt = inst
            ang2 = ab2v(v2, v3)

        if inst_set.size and not _acute(ang) and not _acute(ang2):
            d = int(np.argmin(tempdist))
            link[intt]["x"] = np.concatenate([link[intt]["x"], pointX[d]])
            link[intt]["y"] = np.concatenate([link[intt]["y"], pointY[d]])
            line[intt, 2:4] = [link[intt]["x"][-1], link[intt]["y"][-1]]
            crackRaw[intt] = np.vstack([np.asarray(crackRaw[intt], float), np.asarray(crackRaw[d], float)])
            templine[d] = 0.0
            detind.append(d)
            if d in intt2:
                intt2.remove(d)
        else:
            tempdist = spdist(tail, templine[:, 2:4])       # distance to segment ends
            inst_set = np.flatnonzero(tempdist <= a)
            ang = ang2 = None
            if inst_set.size:
                mn = tempdist[inst_set].min()
                inst = int(np.flatnonzero(tempdist == mn)[0])
                v1 = crackRawt[indt][0] - crackRawt[indt][-1]
                v2 = np.asarray(crackRaw[intt], float)[0] - np.asarray(crackRaw[intt], float)[-1]
                v3 = np.asarray(crackRaw[inst], float)[0] - np.asarray(crackRaw[inst], float)[-1]
                ang = ab2v(v1, v3)
                indt = inst
                ang2 = ab2v(v2, v3)
            if inst_set.size and not _acute(ang) and not _acute(ang2):
                d = int(np.argmin(tempdist))
                link[intt]["x"] = np.concatenate([link[intt]["x"], pointX[d][::-1]])
                link[intt]["y"] = np.concatenate([link[intt]["y"], pointY[d][::-1]])
                line[intt, 2:4] = [link[intt]["x"][-1], link[intt]["y"][-1]]
                crackRaw[intt] = np.vstack([np.asarray(crackRaw[intt], float),
                                            np.asarray(crackRaw[d], float)[::-1]])
                crackRawt[d] = crackRawt[d][::-1]
                templine[d] = 0.0
                detind.append(d)
                if d in intt2:
                    intt2.remove(d)
            else:
                intt = intt2[0]
                indt = intt
                prevx = link[intt]["x"] if link[intt]["x"] is not None else np.empty(0)
                prevy = link[intt]["y"] if link[intt]["y"] is not None else np.empty(0)
                link[intt]["x"] = np.concatenate([np.asarray(prevx, float), pointX[intt]])
                link[intt]["y"] = np.concatenate([np.asarray(prevy, float), pointY[intt]])
                line[intt, 2:4] = [link[intt]["x"][-1], link[intt]["y"][-1]]
                templine[intt] = 0.0
                if intt in intt2:
                    intt2.remove(intt)
        if np.any(templine):
            continue
        break

    # drop merged segments
    detset = sorted(set(detind))
    keep = np.ones(line.shape[0], dtype=bool)
    if detset:
        keep[detset] = False
    line = line[keep]
    crackRaw = [c for c, k in zip(crackRaw, keep) if k]
    fP = fP[keep]
    link = [l for l in link if l["x"] is not None]
    return line, crackRaw, link, fP


def _ImagePlanning_SCC_chord_legacy(img_n, skel=None):
    """Alternative chord-based crack graph (kept for reference).

    Builds the crack graph using endpoint nodes plus a visibility graph over
    Minkowski-buffered crack buffers.  Acute-angle crack segments are held out
    of the connection pass and re-inserted afterward.  The active implementation
    is ``ImagePlanning_SCC`` (taut-string centerline graph, which guarantees
    100 % footprint coverage at minimum cost).

    Parameters
    ----------
    img_n : str or None
        Map name passed to ``_front_end``.
    skel : array-like, optional
        Pre-computed skeleton; bypasses loading and skeletonization.

    Returns
    -------
    node : ndarray, shape (N, 2)
    edgeList : ndarray, shape (E, 2)
    ttt : float
    """
    ttt = 0.0
    a = inpxMap(7 / 2)            # 44
    s = inpxMap(4.5 * 12 / 2)     # 342
    a = a

    BW3 = _front_end(img_n, a, skel=skel)
    yy = BW3.shape[1]

    from private.bwmorph import neighbor_count_points
    eP, _ = neighbor_count_points(BW3, 1)                       # skeleton endpoints
    endP = np.argwhere(np.asarray(eP) > 0)           # (row, col), 0-based
    I = BW3.copy()

    crackRaw, line, pointX, pointY = compCrack(I, endP, _DIR_MAP, None)
    line = np.atleast_2d(np.asarray(line, float)) if len(crackRaw) else np.empty((0, 4))

    # remove cracks shorter than the footprint radius
    c_chq = np.array([total_length(np.asarray(c, float)) < a for c in crackRaw], dtype=bool)
    keep = ~c_chq
    crackRaw = [c for c, k in zip(crackRaw, keep) if k]
    line = line[keep]
    pointX = pointX[keep]
    pointY = pointY[keep]

    link = [{"x": None, "y": None} for _ in range(line.shape[0])]

    # acute-angle pass 1: detect acute-angle segments and hold them out of the connection loop
    fP = np.full(line.shape[0], float(a))
    fP, det = _acute_fP(crackRaw, fP, a, yy, mark_det=True)
    h_line = np.empty((0, 4)); h_crackRaw = []
    h_pointX = np.empty((0, pointX.shape[1] if pointX.size else 24))
    h_pointY = np.empty((0, pointY.shape[1] if pointY.size else 24))
    h_fP = np.empty(0)
    if det:
        dmask = np.zeros(line.shape[0], dtype=bool); dmask[det] = True
        h_line = line[dmask]; h_pointX = pointX[dmask]; h_pointY = pointY[dmask]
        h_crackRaw = [c for c, k in zip(crackRaw, dmask) if k]; h_fP = fP[dmask]
        keep = ~dmask
        line = line[keep]
        crackRaw = [c for c, k in zip(crackRaw, keep) if k]
        link = [l for l, k in zip(link, keep) if k]
        pointX = pointX[keep]; pointY = pointY[keep]; fP = fP[keep]

    # single-pass graph connection
    line, crackRaw, link, fP = _connect_graph_offline(line, crackRaw, link, pointX, pointY, fP, a)

    # re-add held acute-angle edges
    if h_line.shape[0]:
        line = np.vstack([line, h_line]) if line.size else h_line.copy()
        crackRaw = crackRaw + h_crackRaw
        fP = np.concatenate([fP, h_fP]) if fP.size else h_fP.copy()
        for kk in range(h_line.shape[0]):
            link.append({"x": h_pointX[kk][::-1].copy(), "y": h_pointY[kk][::-1].copy()})

    # acute-angle pass 2: update fP only (no detection)
    fP, _ = _acute_fP(crackRaw, fP, a, yy, mark_det=False)

    node, edgeList = _build_graph(line, crackRaw, link, a, s)
    return node, edgeList, ttt


_COVER_FRAC = 0.9     # cover each crack centerline within COVER_FRAC*a (margin for the
#                       thick raw crack + piecewise-linear emission slack) -> 100% fill.


def _walk_component(pts):
    """Trace a connected skeleton component (set of (r,c)) into polylines via 8-adjacency.
    Handles CYCLES (no endpoint/branchpoint -> compCrack skips them) so every pixel is routed."""
    def nbrs(p):
        r, c = p; out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in pts:
                    out.append((r + dr, c + dc))
        return out
    deg = {p: len(nbrs(p)) for p in pts}; used = set(); lines = []
    starts = [p for p in pts if deg[p] == 1] + [p for p in pts if deg[p] != 1]
    for s0 in starts:
        if all(frozenset((s0, q)) in used for q in nbrs(s0)):
            continue
        cur, prev, line = s0, None, [s0]
        while True:
            nxt = [q for q in nbrs(cur) if q != prev and frozenset((cur, q)) not in used]
            if not nxt:
                break
            q = min(nxt, key=lambda z: (z != prev, z)); used.add(frozenset((cur, q)))
            line.append(q); prev, cur = cur, q
            if cur == s0:
                break
        if len(line) >= 2:
            lines.append(np.array(line, float))
    return lines


def _trace_all_segments(BW3):
    """Trace all skeleton pixels into crack polylines, including isolated loops.

    Seeds ``compCrack`` from skeleton endpoints and branch points (column-major order),
    then walks any residual pixels that were not reached by the seeded pass (e.g. pure
    cycles with no endpoint) via 8-adjacency.  Every skeleton pixel is covered.
    """
    from private.bwmorph import endpoints as _ep_lut, branchpoints as _bp_lut
    from scipy.ndimage import label
    from scipy.spatial import cKDTree
    skpx = np.argwhere(np.asarray(BW3) > 0)
    eP = _ep_lut(BW3); bP = _bp_lut(BW3)
    seed = np.argwhere(eP | bP)
    if seed.size:
        seed = seed[np.lexsort((seed[:, 0], seed[:, 1]))]   # column-major scan order
    crackRaw, *_ = compCrack(BW3.copy(), seed, _DIR_MAP, None)
    crackRaw = [np.atleast_2d(np.asarray(c, float)) for c in crackRaw if len(np.atleast_2d(c))]
    traced = np.vstack(crackRaw) if crackRaw else np.empty((0, 2))
    residual = skpx if traced.shape[0] == 0 else skpx[cKDTree(traced).query(skpx, k=1)[0] > 1.5]
    if residual.shape[0]:
        rmask = np.zeros(np.asarray(BW3).shape, bool); rmask[residual[:, 0], residual[:, 1]] = True
        lab, n = label(rmask, structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            crackRaw.extend(_walk_component({(int(r), int(c)) for r, c in np.argwhere(lab == k)}))
    return crackRaw


def _taut_graph(crackRaw, a, cover_frac=_COVER_FRAC, merge_tol=6.0):
    """Each crack segment's centerline -> taut-string boundary min-cover polyline (covered
    within cover_frac*a). Its vertices become graph nodes, consecutive vertices become edges
    (downstream straight-chord emission reproduces the taut polyline). Endpoint/junction
    vertices shared within merge_tol are merged so crossings connect. Returns (node[row,col],
    edgeList[0-based pairs])."""
    from private.min_cover import min_cover_path_adaptive
    reps = []

    def _nid(p):
        p = np.asarray(p, float)
        if reps:
            r = np.asarray(reps); k = int(np.argmin(((r - p) ** 2).sum(1)))
            if ((r[k] - p) ** 2).sum() <= merge_tol ** 2:
                return k
        reps.append((float(p[0]), float(p[1]))); return len(reps) - 1

    edges = []
    for c in crackRaw:
        c = np.atleast_2d(np.asarray(c, float))
        if c.shape[0] < 2:
            continue
        v = min_cover_path_adaptive(c, a * cover_frac)
        ids = [_nid(p) for p in v]
        for k in range(len(ids) - 1):
            if ids[k] != ids[k + 1]:
                edges.append((ids[k], ids[k + 1]))
    node = np.array(reps, float) if reps else np.empty((0, 2))
    if not edges:
        return node, np.empty((0, 2), int)
    seen = set(); uniq = []
    for u, v in edges:
        key = (u, v) if u < v else (v, u)
        if key not in seen:
            seen.add(key); uniq.append((u, v))
    return node, np.array(uniq, int)


def ImagePlanning_SCC(img_n, skel=None):
    """Build the crack graph for the SCC offline planner (taut-string centerline method).

    Skeletonizes the crack map, traces all crack polylines, and constructs a minimum-
    length centerline graph whose straight chords cover every crack within the footprint
    radius, guaranteeing 100 % footprint coverage at minimum path cost.

    Parameters
    ----------
    img_n : str
        Map identifier.  Uniform maps are loaded from ``CrackMaps/Uniform/<img_n>.png``;
        Gaussian maps from the corresponding ``.mat`` file.
    skel : array-like, optional
        Pre-computed binary skeleton array; bypasses loading and skeletonization.

    Returns
    -------
    node : ndarray, shape (N, 2)
        Crack-graph node coordinates in ``(row, col)`` order.
    edgeList : ndarray, shape (E, 2)
        Undirected node-index pairs, 0-based.
    ttt : float
        Timing placeholder (always 0.0).
    """
    ttt = 0.0
    a = inpxMap(7 / 2)            # 44
    BW3 = _front_end(img_n, a, skel=skel)
    crackRaw = _trace_all_segments(BW3)
    node, edgeList = _taut_graph(crackRaw, a, cover_frac=_COVER_FRAC)
    return node, edgeList, ttt


def _build_graph(line, crackRaw, link, a, s):
    """Build the node/edge visibility graph from connected crack segments.

    Steps: collect endpoint nodes -> Minkowski-buffer each segment by ``a`` ->
    find pairwise overlap centroids -> merge candidates within ``a`` -> shorten
    endpoints to the nearest skeleton point within ``a/√2`` -> build visibility
    graph (line-of-sight test, pathfinder for occluded pairs).

    Parameters
    ----------
    line : ndarray, shape (K, 4)
        Each row ``[x0, y0, x1, y1]`` (start/end coords of a crack segment).
    crackRaw : list of ndarray
        Crack polylines in ``(row, col)`` order.
    link : list of dict
        Per-segment dicts with keys ``'x'`` and ``'y'`` (linked centerline coords).
    a : float
        Footprint radius in pixels.
    s : float
        Sensor radius in pixels.

    Returns
    -------
    node : ndarray, shape (N, 2)
    edgeList : ndarray, shape (E, 2)
    """
    K = line.shape[0]
    if K == 0:
        return np.empty((0, 2)), np.empty((0, 2), int)

    # ---- endpoint nodes ----
    starts = line[:, [0, 1]]; ends = line[:, [2, 3]]
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

    # ---- Minkowski buffer: expand each link polyline by radius `a` ----
    S = []; v = []
    for i in range(K):
        lx = np.asarray(link[i]["x"], float); ly = np.asarray(link[i]["y"], float)
        poly = pu.polybuffer(np.column_stack([ly, lx]), a, kind="lines")   # (x=col,y=row)
        bd = poly.boundary()
        px, py = bd[:, 0], bd[:, 1]
        dec = DecimatePoly(np.column_stack([px, py]), [1, 1], False)
        if isinstance(dec, tuple):
            dec = dec[0]
        dec = np.asarray(dec, float)
        poly = pu.polyshape(dec[:, 0], dec[:, 1])
        S.append(poly); v.append(poly)

    # ---- pairwise overlap -> candidate nodes ----
    overlap = False
    for i in range(K):
        for j in range(K):
            if i != j and v[i].geom is not None and v[j].geom is not None:
                if v[i].geom.intersection(v[j].geom).area > 1e-9:
                    overlap = True; break
        if overlap:
            break
    if overlap:
        Geo = Polygons_Intersection(S, 0, 1e-3)
        Ind = []; nodes = []
        for g in Geo:
            if len(g["index"]) >= 2:
                region = g["P"]
                parts = list(region.geoms) if region.geom_type.startswith("Multi") else [region]
                for part in parts:
                    Ind.append(len(g["index"]))
                    c = part.centroid
                    nodes.append([c.y, c.x])          # [cy,cx]->[cx,cy]=(row,col)
        nodes = np.array(nodes) if nodes else np.empty((0, 2))
        if nodes.shape[0]:
            order = np.argsort(-np.asarray(Ind), kind="stable")
            SortNode = nodes[order]
        else:
            SortNode = np.empty((0, 2))
        NodeCan = np.vstack([SortNode, endNodes]) if (SortNode.size or endNodes.size) else np.empty((0, 2))
        node = []
        for nc in NodeCan:
            if not node or np.all(spdist(nc, np.array(node)) > a):
                node.append(nc)
        node = np.array(node) if node else np.empty((0, 2))
    else:
        node = endNodes.copy()

    # ---- shorten endpoints to nearest skeleton point within footprint/√2 ----
    for e in range(node.shape[0]):
        aRan = pu.polybuffer(node[e], a / sqrt(2), kind="points")
        logi = np.array([_intersect_nonempty(aRan, c) for c in crackRaw], dtype=bool)
        if logi.sum() == 1:
            ci = int(np.flatnonzero(logi)[0])
            cr = np.asarray(crackRaw[ci], float)
            if np.any(pu.isinterior(aRan, cr[[0, -1]])):
                out = _clip_outside(aRan, cr)
                if out.shape[0]:
                    crackRaw[ci] = out
                    ee_pts = out[[0, -1]]
                    node[e] = ee_pts[int(np.argmin(spdist(node[e], ee_pts)))]

    # ---- visibility graph ----
    vgNE = [None] * len(S); vgEE = [None] * len(S)
    for i in range(len(S)):
        Vx = S[i].Vertices[:, 0]; Vy = S[i].Vertices[:, 1]
        IN, _ = pu.inpolygon(node[:, 0], node[:, 1], Vy, Vx)
        if not IN.any():
            vgNE[i] = np.empty((0, 4)); vgEE[i] = np.empty((0, 2)); continue
        insidx = np.flatnonzero(IN)
        VP = np.column_stack([insidx, node[IN]])
        linkpts = np.column_stack([np.asarray(link[i]["x"], float), np.asarray(link[i]["y"], float)])
        mem = _ismember_rows(VP[:, 1:3], linkpts)
        _, dsr = _dsearchn(linkpts, VP[:, 1:3])
        mem = mem | (dsr < a)
        if mem.any():
            ks, _ = _dsearchn(linkpts, VP[mem, 1:3])
            tab = np.column_stack([ks, VP[mem]])
            tab = tab[np.argsort(tab[:, 0], kind="stable")]
        else:
            tab = np.empty((0, 4))
        vne = []; vee = []
        for j in range(tab.shape[0] - 1):
            vne.append([tab[j, 2], tab[j, 3], tab[j + 1, 2], tab[j + 1, 3]])
            vee.append([tab[j, 1], tab[j + 1, 1]])
        if (~mem).any() and tab.shape[0]:
            t2, _ = _dsearchn(tab[:, 2:4], VP[~mem, 1:3])
            for q, nidx in zip(np.flatnonzero(~mem), t2):
                vne.append([VP[q, 1], VP[q, 2], tab[nidx, 2], tab[nidx, 3]])
                vee.append([VP[q, 0], tab[nidx, 1]])
        vgNE[i] = np.array(vne) if vne else np.empty((0, 4))
        vgEE[i] = np.array(vee) if vee else np.empty((0, 2))

    for i in range(len(S)):
        if vgNE[i].shape[0] == 0:
            continue
        bnd = np.vstack([S[i].Vertices[:, 1], S[i].Vertices[:, 0]])
        vis = line_of_sight_poly(vgNE[i][:, 0:2], vgNE[i][:, 2:4], bnd)
        for j in range(len(vis)):
            if not vis[j]:
                start = vgNE[i][j, 0:2]; goal = vgNE[i][j, 2:4]
                boundary = np.column_stack([S[i].Vertices[:, 1], S[i].Vertices[:, 0]])
                startn = vgEE[i][j, 0]; goaln = vgEE[i][j, 1]
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
        mask = np.ones(vgEE[i].shape[0], dtype=bool)
        mask[:keepe.shape[0]] = keepe
        vgNE[i] = vgNE[i][mask]; vgEE[i] = vgEE[i][mask]

    edgeList = np.empty((0, 2))
    for i in range(len(vgEE)):
        if vgEE[i] is not None and vgEE[i].size:
            edgeList = np.vstack([edgeList, vgEE[i]]) if edgeList.size else np.asarray(vgEE[i], float)
    edgeList = edgeList.astype(int)
    return node, edgeList

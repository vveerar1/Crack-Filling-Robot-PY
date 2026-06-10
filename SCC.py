"""Offline Sensor-based Complete Coverage (SCC) planner.

SCC plans one path for a mobile crack-filling robot to (1) completely cover a
known work area with its sensor and (2) fill every crack in it, at low travel
cost. Offline / known environment: the crack map is given up front, so the whole
route is computed before the robot moves. (OnlineSCC is the counterpart that
discovers cracks while scanning.)

Flow
  1. Crack graph    Extract each crack from the map and turn it into a graph of
                    fill waypoints (ImagePlanning_SCC).
  2. Free space     Buffer the cracks by the sensor radius and subtract them from
                    the work area, leaving the region the robot must cover.
  3. Decomposition  Split that region at its critical points into monotone
                    coverage cells (Morse Cell Decomposition, MCD) and build the
                    cell-adjacency graph (Reeb).
  4. Route + sweep  Order the cells over a graph (a Rural-/Chinese-Postman tour)
                    and lay a boustrophedon (zig-zag) sweep through each, joined
                    with the crack-fill path, into one closed coverage+fill tour.

Route options (the ``route`` argument, default "rpp"):
  "rpp"     geometry-routed cell Rural-Postman tour            [default]
  "best"    cheapest of several formulations, coverage-guarded
  "native"  cell-adjacency-graph Chinese-Postman tour
  "matlab"  cell-adjacency-graph Chinese-Postman tour, MATLAB cell order

Robot model: the base / footprint / sensor diameters (configured in
``robot_config.json``) give the base reach r1, the fill-footprint radius a and
the sensor radius s; the zig-zag spacing is the square inscribed in the sensor
circle (s/sqrt2).

Entry point: ``run_scc(...) -> (PathEdge, res)``. Run ``python SCC.py`` to
select a map and plan it; see ``--help``.
"""

import math
import os
import time

import numpy as np
from scipy.spatial.distance import pdist, squareform

from private import poly_utils as pu
from private.utils import inpxMap, pxinMap, mfix, spdist, spdist2, total_length
from MCD import MCD
from Reeb import Reeb
from Boustrophedon_CellCon import Boustrophedon_CellCon
from ChinesePostman import ChinesePostman
from ImagePlanning_SCC import ImagePlanning_SCC
from OnlineSCC import _regJoin, _reg_list
from private.run_log import LOG, rotate_path
from private.config_loader import ROBOT1

_DEN = [35, 45, 50, 65, 80, 90, 95, 100]

# Critical-ordering policy for the cell graph. "native" = the MCD accumulation order.
# Other keys ("l2", "colrow", "angle") and "best" (cheapest of _BEST_KEYS) are
# experimental alternatives selectable via the OSCC_CRIT_ORDER env variable.
_CRIT_ORDER_KEY = os.environ.get("OSCC_CRIT_ORDER", "native")
_BEST_KEYS = ("native", "l2", "colrow", "angle")
# Optional override: (N,2) array of critical-point coords (row,col); when set, run_scc
# reorders subcritP to match that target order (coord-nearest) before building the graph.
_FORCE_ORDER = None
# SCC route formulation. Default "rpp" = the geometry-routed celled-rpp (single deterministic,
# order-invariant pass; cheapest single formulation and 100% coverage + fill on all maps).
# "best" = coverage-guarded best-of-3; "native"/"matlab" = reeb ChinesePostman (two cell orderings).
# Per-call override: run_scc(..., route="...").
_SCC_ROUTE_MODE = os.environ.get("OSCC_SCC_ROUTE", "rpp")     # "rpp" | "best" | "native" | "matlab"
_ROUTE_ALIASES = {"off": "native", "on": "matlab"}           # backward-compatible alias strings


def _norm_route(r):
    """Normalize a route-mode string; resolve the legacy 'off'/'on' alias names."""
    return _ROUTE_ALIASES.get(r, r)
# Optional override: (N,2) array (row,col); when set, run_scc snaps each allNode row
# to the nearest target coord before computing Dist/ChinesePostman.
_FORCE_ALLNODE = None


def _polyclean_scc(polyin):
    """Drop coverage regions with area*1e-3 <= 10 (integer-truncated), join the rest."""
    regs = _reg_list(polyin)
    regs = [r for r in regs if mfix(r.area * 1e-3) > 10]
    return _regJoin(regs) if regs else pu.polyshape([], [])


def _prune_rMiss_only(subcritP, splitReg_work, reebEdge, s):
    """Merge each critical point not referenced by any Reeb edge into its nearest
    neighbour's cell (bare union, no +/-1 buffer), rewire reebEdge, then delete
    the orphaned critical points and remap the surviving reebEdge indices."""
    subcritP = np.atleast_2d(np.asarray(subcritP, float)).copy()
    reebEdge = np.atleast_2d(np.asarray(reebEdge, int)).copy()
    splitReg_work = list(splitReg_work)
    n = subcritP.shape[0]

    def _uniq(re):
        return np.unique(re) if re.size else np.empty(0, int)

    rMiss = [i for i in range(n) if i not in set(_uniq(reebEdge).tolist())]
    r_rmin = []
    indj = []
    rmin = {}
    for ii, i in enumerate(rMiss):
        rdis = spdist(subcritP[i], subcritP)
        rdis[rdis == 0] = np.inf
        rmin[i] = int(np.argmin(rdis))
        if np.any(rdis < s):
            # rows of reebEdge that contain rmin[i]
            contain = np.flatnonzero((reebEdge == rmin[i]).any(axis=1)) if reebEdge.size else np.empty(0, int)
            # edges sharing any node with those rows
            if contain.size:
                touched_nodes = np.unique(reebEdge[contain])
                rAreaN = np.flatnonzero(np.isin(reebEdge, touched_nodes).any(axis=1))
            else:
                rAreaN = np.empty(0, int)
            if np.size(np.flatnonzero((reebEdge == rmin[i]).any(axis=1))) < 3:
                rAreaN = np.flatnonzero((reebEdge == rmin[i]).any(axis=1))
                if rAreaN.size:
                    rNodeN = reebEdge[rAreaN]
                    if rNodeN.shape[0] > 1:
                        reebEdge = np.delete(reebEdge, rAreaN, axis=0)
                        r_rmin.append(rmin[i])
                        rArea = splitReg_work[rAreaN[0]]
                        for k in rAreaN[1:]:
                            rArea = pu.union(rArea, splitReg_work[k])
                        for k in sorted(rAreaN, reverse=True):
                            del splitReg_work[k]
                        splitReg_work.append(rArea)
                        rest = rNodeN[rNodeN != rmin[i]]
                        if rest.size:
                            reebEdge = np.vstack([reebEdge, rest.reshape(1, -1)]) if reebEdge.size else rest.reshape(1, -1)
        else:
            indj.append(ii)
    for ii in sorted(indj, reverse=True):
        del rMiss[ii]

    # delete orphaned subcritP rows + remap reebEdge
    used = set(_uniq(reebEdge).tolist())
    notused = [i for i in range(subcritP.shape[0]) if i not in used]
    drop = sorted(set(notused) | set(rMiss) | set(r_rmin))
    keep = [i for i in range(subcritP.shape[0]) if i not in set(drop)]
    remap = {old: new for new, old in enumerate(keep)}
    subcritP = subcritP[keep] if keep else np.empty((0, 2))
    if reebEdge.size:
        reebEdge = np.array([[remap.get(int(u), -1), remap.get(int(v), -1)] for u, v in reebEdge])
        reebEdge = reebEdge[(reebEdge >= 0).all(axis=1)] if reebEdge.size else reebEdge
    return subcritP, splitReg_work, reebEdge


# Drop leaf coverage cells whose short dimension is narrower than s - s/sqrt2 (~100px)
# and that have no crack nodes inside; those cells are already sensed by adjacent sweeps.
# DISABLED: a thin leaf cell can still be the cheapest corridor to the far side of the
# map, so dropping it regresses the reconnection cost. Revisit after route optimisation.
_DROP_THIN_CELLS = False


def _drop_redundant_thin_cells(splitReg_work, reebEdge, reebCell, node, s):
    from shapely.geometry import Point
    thr = s - s / math.sqrt(2)
    reebEdge = np.atleast_2d(np.asarray(reebEdge, int)).copy()
    reebCell = np.asarray(reebCell, int).copy()
    splitReg_work = list(splitReg_work)
    nodes = np.atleast_2d(np.asarray(node, float)) if np.size(node) else np.empty((0, 2))
    # reeb-node degree: a cell is a DEAD-END (pendant) iff its Reeb edge has a degree-1
    # endpoint. Only dead-end thin cells are wasted out-and-back excursions; a thin cell
    # that is a through-CORRIDOR (both endpoints degree>=2) is the cheapest transit to the
    # far side, so dropping it forces a LONGER detour.
    deg = {}
    for u, v in reebEdge:
        deg[int(u)] = deg.get(int(u), 0) + 1
        deg[int(v)] = deg.get(int(v), 0) + 1
    drop = []
    for k, c in enumerate(splitReg_work):
        g = c.geom if hasattr(c, "geom") else c
        if g is None or g.is_empty:
            continue
        minx, miny, maxx, maxy = g.bounds
        if min(maxx - minx, maxy - miny) > thr:
            continue                                            # not thin
        rows = np.flatnonzero(reebCell == k) if reebCell.size else np.empty(0, int)
        if rows.size == 0:
            continue
        is_leaf = any(deg.get(int(u), 0) == 1 or deg.get(int(v), 0) == 1
                      for u, v in reebEdge[rows])
        if not is_leaf:
            continue                                            # through-corridor: keep
        has_crack = any(g.contains(Point(float(nx), float(ny))) for ny, nx in nodes)  # node=(row,col)
        if not has_crack:
            drop.append(k)
    if not drop:
        return splitReg_work, reebEdge, reebCell
    dropset = set(drop)
    if reebCell.size:
        keep = np.array([int(reebCell[i]) not in dropset for i in range(reebCell.size)])
        reebEdge = reebEdge[keep] if reebEdge.size else reebEdge
        reebCell = reebCell[keep]
    splitReg_work = [c for k, c in enumerate(splitReg_work) if k not in dropset]
    drop_arr = np.array(sorted(drop))
    if reebCell.size:
        reebCell = np.array([int(c) - int((drop_arr < c).sum()) for c in reebCell], int)
    return splitReg_work, reebEdge, reebCell


# Frederickson RPP pre-augmentation: connect the disconnected required-edge components
# by MST over component closest-pairs, then optionally min-match the odd-degree nodes,
# so ChinesePostman receives a connected (optionally Eulerian) graph. DISABLED: the
# route cost is dominated by boustrophedon cell-sweep order, not reconnect-edge length,
# so pre-augmentation regresses the actual tour even as it reduces graph weight.
_RPP_RECONNECT = False
_RPP_MODE = "mst"     # "mst" = connect components only; "full" = also pre-match -> Eulerian


def _rpp_augment(allEdge, allNode):
    """Return Frederickson augmentation (u,v) node-index pairs that make the required-edge
    graph connected + Eulerian at near-minimum added length. [] if already connected."""
    import networkx as nx
    from itertools import combinations
    P = np.atleast_2d(np.asarray(allNode, float))
    N = P.shape[0]
    G = nx.MultiGraph()
    G.add_nodes_from(range(N))
    for u, v in np.atleast_2d(allEdge):
        G.add_edge(int(u), int(v))
    comps = [c for c in nx.connected_components(G) if len(c) > 1]
    if len(comps) <= 1:
        return []                          # connected -> ChinesePostman's MILP match is optimal
    d = lambda u, v: float(np.hypot(*(P[u] - P[v])))
    aug = []
    # 1) connect components: MST over components, weight = closest node-pair distance
    CG = nx.Graph()
    for a, b in combinations(range(len(comps)), 2):
        u, v = min(((u, v) for u in comps[a] for v in comps[b]), key=lambda p: d(*p))
        CG.add_edge(a, b, weight=d(u, v), pair=(int(u), int(v)))
    for a, b in nx.minimum_spanning_tree(CG).edges:
        u, v = CG[a][b]["pair"]
        aug.append((u, v)); G.add_edge(u, v)
    # 2) min-weight perfect matching of the odd-degree nodes (skipped in 'mst' mode, where we
    #    only connect the components and let ChinesePostman's MILP do the matching + ordering)
    if _RPP_MODE == "full":
        odd = [n for n in G.nodes if G.degree(n) % 2 == 1]
        if odd:
            MG = nx.Graph()
            for u, v in combinations(odd, 2):
                MG.add_edge(u, v, weight=-d(u, v))
            for u, v in nx.max_weight_matching(MG, maxcardinality=True):
                aug.append((int(u), int(v)))
    return aug


def run_scc(img_n="myCrack8_100_1", dd=8, skel=None, _node=None, _edgeList=None,
            viz=None, out=None, _out=None, crit_order=None, matlab_start=None, route=None):
    """Plan the offline SCC coverage+fill tour for one crack map.

    Parameters
      img_n : crack-map name under CrackMaps/ -- a Uniform map, or a Gaussian
              'Gaussian<b>/myCrackGauss_s<sig>_<den>' name.
      dd    : crack-density index (1..8); used only to label the result row.
      route : route formulation -- "rpp" (default), "best", "native" or "matlab"
              (see the module docstring). Defaults to env OSCC_SCC_ROUTE, else "rpp".
      viz   : "show" opens a result figure; "save" (or a path via ``out``) writes the
              final-path plot; "live" redraws the figure at each pipeline stage
              (workspace -> MCD -> Reeb -> route -> path; needs a GUI backend).
              The final-path PNG is written to Results/SCC/ in every mode.
      out   : output-path override (.png -> the final plot, .gif -> the animation).

    Returns
      PathEdge : (M, 2) coverage+fill path as (x, y) points.
      res      : [coverage %, density %, run-time s, path length ft, area covered sq ft].
    """
    _mode = _norm_route(route if route is not None else _SCC_ROUTE_MODE)
    # robot dimensions from robot_config.json (edit there to reconfigure the robot)
    botD = ROBOT1.base_diameter_in          # small base, so the nozzle can reach near-wall cracks
    footD = ROBOT1.footprint_diameter_in
    sensD = ROBOT1.sensor_diameter_in
    r1, a, s = ROBOT1.r1, ROBOT1.a, ROBOT1.s   # base / footprint / sensor radius (px)
    crack_reach = r1 - a          # base-reach limit for crack-fill (decoupled from s/sqrt2)
    smflag = 0

    # Best-of-K critical ordering: try K deterministic geometric orderings of the critical
    # points and keep the cheapest route. The crack-graph front-end is shared; each key
    # re-runs only MCD -> route.
    if crit_order is None and _CRIT_ORDER_KEY == "best":
        if _node is None or _edgeList is None:
            if skel is None:
                from ImagePlanning_SCC import _front_end
                skel = _front_end(img_n, a)
            _node, _edgeList, _ = ImagePlanning_SCC(img_n, skel=skel)
        best = best_key = None
        for _k in _BEST_KEYS:
            _P, _res = run_scc(img_n, dd, _node=_node, _edgeList=_edgeList, crit_order=_k)
            if best is None or _res[3] < best[1][3]:
                best, best_key = (_P, _res), _k
        if viz is not None or out is not None or _out is not None:   # render the winning route
            return run_scc(img_n, dd, _node=_node, _edgeList=_edgeList,
                           viz=viz, out=out, _out=_out, crit_order=best_key)
        return best

    # ---- coverage-guarded best-of-2 route -- the DEFAULT ----
    # Run the two deterministic cell-ordering formulations (native MCD order and the
    # boundary-scan order), share the crack-graph front-end, and keep the cheaper route
    # whose coverage does not regress more than 0.5 percentage points below the other.
    # Only a top-level call (matlab_start is None) enters this block.
    if matlab_start is None and crit_order is None and _mode == "best":
        if _node is None or _edgeList is None:
            if skel is None:
                from ImagePlanning_SCC import _front_end
                skel = _front_end(img_n, a)
            _node, _edgeList, _ = ImagePlanning_SCC(img_n, skel=skel)
        cand = []                                             # (tag, PathEdge, res, out_dict)
        rpp_in = None
        for _msflag in (False, True):                         # candidates 1-2: reeb ChinesePostman
            _o = {}
            _P, _res = run_scc(img_n, dd, _node=_node, _edgeList=_edgeList,
                               matlab_start=_msflag, _out=_o)
            cand.append((_msflag, _P, _res, _o))
            if rpp_in is None:
                rpp_in = _o.get("_rpp")
        # candidate 3: geometry-routed celled-rpp (order-invariant; cells shared).
        # Scoring only -- viz renders the winning reeb candidate.
        if rpp_in is not None and viz is None and out is None:
            try:
                from private import celled_rpp as _CR
                _pe = _CR.route(rpp_in["node"], rpp_in["crackEdge"], rpp_in["cell_geoms"],
                                rpp_in["cell_verts"], rpp_in["s"], rpp_in["a"])
                if np.atleast_2d(_pe).shape[0] >= 2:
                    _pe = np.asarray(_pe, float)
                    _sl = pxinMap(total_length(_pe)) / 12
                    _cov = pu.polybuffer(_pe, s / math.sqrt(2), kind="lines").area / (2898 * 3050) * 100
                    _ac = 2 * pxinMap(s / math.sqrt(2)) / 12 * _sl
                    cand.append(("rpp", _pe, [_cov, _DEN[dd - 1], 0.0, _sl, _ac], {}))
            except Exception:
                pass
        # Coverage-guarded best-of-N: reeb candidates are always eligible; celled-rpp joins
        # only if its coverage does not regress below the reeb baseline (cov >= min reeb - 0.5).
        # Floor is on the reeb baseline (not max coverage) to avoid evicting a cheaper reeb route.
        reeb_cov = [c[2][0] for c in cand if c[0] in (False, True)]
        floor = (min(reeb_cov) - 0.5) if reeb_cov else -1e9
        elig = [c for c in cand if c[0] in (False, True) or c[2][0] >= floor] or cand
        chosen = min(elig, key=lambda c: c[2][3])
        if viz is not None or out is not None:                # render the winning reeb route
            return run_scc(img_n, dd, _node=_node, _edgeList=_edgeList,
                           matlab_start=chosen[0], viz=viz, out=out, _out=_out)
        if _out is not None:
            _out.update(chosen[3])
        return chosen[1], chosen[2]

    # run logging (no-op unless enabled). __main__ turns it on; tests/sweeps leave it off.
    # Set env SCC_DRV_DBG=1 to force console logging.
    if not LOG.enabled and os.environ.get("SCC_DRV_DBG"):
        LOG.enable(console=True)
    LOG.section(f"SCC  |  map: {img_n if isinstance(img_n, str) else '<crackGen>'}")
    LOG.kv("algorithm", "SCC (offline, crack map known up front)")
    LOG.kv("robot base", f"{botD}in dia  ->  r1 = {r1} px")
    LOG.kv("footprint", f"{footD}in dia  ->  a = {a} px")
    LOG.kv("sensor", f"{sensD:.0f}in dia  ->  s = {s} px   (zig-zag spacing {s / math.sqrt(2):.1f} px)")
    LOG.kv("crack-fill reach", f"r1 - a = {crack_reach} px  (near-wall fill; sweep stays s/sqrt2)")
    rowBW, colBW = 2896, 3048
    WSarea = 572.635

    if _node is not None and _edgeList is not None:
        node = np.atleast_2d(np.asarray(_node, float))
        edgeList = np.atleast_2d(np.asarray(_edgeList, int))
        ttt = 0.0
    else:
        if viz is not None and skel is None:          # compute the skeleton ONCE (reused by viz)
            from ImagePlanning_SCC import _front_end
            skel = _front_end(img_n, a)
        node, edgeList, ttt = ImagePlanning_SCC(img_n, skel=skel)
    crackEdge = np.atleast_2d(np.asarray(edgeList, int))   # 0-based into node
    LOG.sub("Crack graph (taut-string)")
    LOG.kv("crack nodes / edges", f"{np.atleast_2d(node).shape[0]} / "
           f"{crackEdge.shape[0] if crackEdge.size else 0}"
           + ("   (pre-computed)" if _node is not None else ""))

    _dbg = bool(os.environ.get("SCC_DBG"))
    _t0 = time.time()
    def _stage(msg):
        if _dbg:
            print(f"[scc {time.time()-_t0:7.1f}s] {msg}", flush=True)

    # live-debug figure (viz mode "live"): redraws at each pipeline stage. No-op headless.
    _live = None
    if viz is not None:
        from private.visualize import VizOptions as _VO, SCCLiveDebug as _SLD
        if (viz.mode if isinstance(viz, _VO) else viz) == "live":
            _live = _SLD(title=f"SCC — {img_n if isinstance(img_n, str) else 'crackGen'}",
                         rowBW=rowBW, colBW=colBW,
                         style=(viz.style if isinstance(viz, _VO) else "color"))

    # objCrack: union of each crack edge's stadium buffer (radius s), in (x,y)
    objs = []
    for (u, v) in crackEdge:
        pts = np.array([node[u], node[v]], float)[:, ::-1]   # (row,col)->(x,y)
        objs.append(pu.polybuffer(pts, s, kind="lines"))
    objCrack = objs[0]
    for o in objs[1:]:
        objCrack = pu.union(objCrack, o)

    # workspace
    Y = [0, colBW, colBW, 0]
    X = [0, 0, rowBW, rowBW]
    extBound = pu.polyshape(Y, X)
    final = extBound
    final_ws = pu.subtract(final, objCrack)
    final_ws = _polyclean_scc(final_ws)
    final_work = final_ws
    _stage('objCrack+workspace done')
    if _live:
        _live.update("workspace + crack graph", skel=skel, node=node, crackEdge=crackEdge)
    critP = 0
    PathEdge = np.empty((0, 2))

    # ---- MCD (working = final_ws directly; NO morphological open) ----
    working = final_ws
    _ms = (_mode == "matlab") if matlab_start is None else bool(matlab_start)
    subcritP, splitReg_work, splitReg, splitEdge = MCD(working, final_work, final_ws, critP, smflag,
                                                       matlab_start=_ms)
    subcritP = np.atleast_2d(subcritP)[:, ::-1]              # MCD returns (x,y); convert to (row,col)
    splitReg_work = _polyclean_scc(splitReg_work)
    if not isinstance(splitReg_work, list):
        splitReg_work = _reg_list(splitReg_work)
    _stage(f'MCD done: critP={np.atleast_2d(subcritP).shape[0]} cells={len(splitReg_work)}')
    if _live:
        _live.update("MCD: cells + critical points", cells=list(splitReg_work), critP=subcritP)
    LOG.sub("Morse cell decomposition + Reeb")
    LOG.kv("MCD", f"critical points: {np.atleast_2d(subcritP).shape[0]}   "
           f"coverage cells: {len(splitReg_work)}")

    # ---- Reeb (first pass) + rMiss prune ----
    reebEdge, _, _, _, _ = Reeb(splitReg_work, subcritP, splitEdge)
    reebEdge = np.atleast_2d(np.asarray(reebEdge, int))
    subcritP, splitReg_work, reebEdge = _prune_rMiss_only(subcritP, splitReg_work, reebEdge, s)
    _stage(f'Reeb1+prune done: critP={np.atleast_2d(subcritP).shape[0]} reebEdge={np.atleast_2d(reebEdge).shape[0]}')

    reebEdge, reebCell, reeb, reebwall, remreg = Reeb(splitReg_work, subcritP, splitEdge)
    reebEdge = np.atleast_2d(np.asarray(reebEdge, int))
    reebCell = np.asarray(reebCell, int)
    if len(remreg):
        rem = np.atleast_1d(np.asarray(remreg, int))
        splitReg_work = [c for k, c in enumerate(splitReg_work) if k not in set(rem.tolist())]
        # remap reebCell to the compacted splitReg_work indexing: removing remreg cells
        # shifts every later cell down by the count of removed cells below it. Without
        # this, reebCell keeps stale (too-large) indices and Boustrophedon_CellCon
        # raises an out-of-range error.
        if reebCell.size:
            reebCell = np.array([int(c) - int((rem < c).sum()) for c in reebCell], int)

    # ---- drop redundant thin coverage cells ----
    if _DROP_THIN_CELLS:
        n0 = len(splitReg_work)
        splitReg_work, reebEdge, reebCell = _drop_redundant_thin_cells(
            splitReg_work, reebEdge, reebCell, node, s)
        _stage(f'thin-cell drop: {n0} -> {len(splitReg_work)} cells')

    # ---- geometry-routed celled-rpp as the SOLE route (OSCC_SCC_ROUTE="rpp", DEFAULT) ----
    # A single deterministic, order-invariant pass on the MCD cells: cheapest single formulation
    # + 100% coverage & fill. Skips the reeb ChinesePostman + Boustrophedon route below.
    # Top-level only (recursive sub-calls keep the reeb route).
    if matlab_start is None and crit_order is None and _mode == "rpp" and len(splitReg_work):
        from private import celled_rpp as _CR
        _cg = [c.geom for c in splitReg_work]
        _cv = [np.atleast_2d(np.asarray(c.Vertices, float)) for c in splitReg_work]
        _pe = np.atleast_2d(_CR.route(np.asarray(node, float), crackEdge, _cg, _cv, float(s), float(a)))
        if _pe.shape[0] >= 2:
            PathEdge = _pe
            scclen = pxinMap(total_length(PathEdge)) / 12
            areaCover = 2 * pxinMap(s / math.sqrt(2)) / 12 * scclen
            areaCover1 = pu.polybuffer(PathEdge, s / math.sqrt(2), kind="lines").area / (2898 * 3050) * 100
            res = [areaCover1, _DEN[dd - 1], ttt, scclen, areaCover]
            LOG.section("RESULT (celled-rpp)")
            LOG.kv("path length", f"{scclen:.3f} ft")
            if _out is not None:
                _out.update({"Path": [], "n_node": int(np.atleast_2d(node).shape[0]),
                             "n_edge": int(crackEdge.shape[0]), "WSarea": WSarea,
                             "coverPercent": (areaCover / WSarea) if WSarea else float("nan")})
            if _live is not None:
                _live.update("Reeb decomposition", cells=list(splitReg_work), critP=subcritP, reeb=reeb)
                _live.update("celled-rpp coverage path", path=PathEdge, clear=("reeb",))
                _live.animate_robot(PathEdge, s, a)          # robot sweeps the full path
                _live.hold()
            if viz is not None or out is not None:           # ALWAYS write the final-path PNG (+ GIF for save)
                _scc_viz(viz, img_n, out, res, cells=list(splitReg_work), subcritP=subcritP,
                         reeb=reeb, node=node, crackEdge=crackEdge, add=np.empty((0, 2), int),
                         allNode=np.asarray(node, float), PathEdge=PathEdge, skel=skel, a=a)
            return PathEdge, res
        # celled-rpp produced nothing usable -> fall through to the reeb route as a safety net

    # ---- ORDER the critical points ----
    # The MCD accumulation order can yield a suboptimal ChinesePostman tour on certain maps.
    # Re-sorting subcritP by a deterministic geometric key can reach a shorter route.
    # _FORCE_ORDER overrides with an explicit target order; OSCC_CRIT_ORDER selects the sort key.
    if subcritP.shape[0] > 1:
        if _FORCE_ORDER is not None:
            tgt = np.atleast_2d(np.asarray(_FORCE_ORDER, float))   # (row,col) in desired order
            used, perm = set(), []
            for t in tgt:
                d = np.hypot(subcritP[:, 0] - t[0], subcritP[:, 1] - t[1])
                for idx in np.argsort(d):
                    if int(idx) not in used:
                        used.add(int(idx)); perm.append(int(idx)); break
            order = np.array(perm, int)
        else:
            key = crit_order if crit_order is not None else _CRIT_ORDER_KEY
            x, y = subcritP[:, 1], subcritP[:, 0]                  # col = x, row = y
            if key == "native":
                order = np.arange(subcritP.shape[0])
            elif key == "colrow":
                order = np.lexsort((y, x))
            elif key == "angle":
                order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
            else:                                                  # "l2"
                order = np.lexsort((y, x, np.hypot(x, y)))         # primary L2, then col, then row
        if order.size == subcritP.shape[0]:
            inv = np.argsort(order); subcritP = subcritP[order]
            if np.size(reebEdge):
                reebEdge = inv[np.asarray(reebEdge, int)]

    # ---- combine crack + Reeb graphs ----
    allNode = np.vstack([node, subcritP]) if subcritP.size else node.copy()
    reebEdge_off = reebEdge + node.shape[0] if reebEdge.size else reebEdge   # offset past crack nodes
    # GCC decision uses the MCD coverage-cell count (splitReg_work), not the raw
    # workspace region count. An interior crack leaves the workspace topologically
    # one region even when MCD decomposes it into multiple coverable cells, so using
    # regions(final_ws) would wrongly skip full boustrophedon coverage on those maps.
    GCC = 0
    if len(splitReg_work) <= 1:
        allNode = node.copy()
        GCC = 1

    if GCC:
        allEdge = crackEdge.copy()
    else:
        allEdge = np.vstack([crackEdge, reebEdge_off]) if reebEdge_off.size else crackEdge.copy()

    if _FORCE_ALLNODE is not None:                    # snap allNode coords to target
        tgt = np.atleast_2d(np.asarray(_FORCE_ALLNODE, float))
        snapped = allNode.astype(float).copy()
        for i in range(snapped.shape[0]):
            d = np.hypot(tgt[:, 0] - snapped[i, 0], tgt[:, 1] - snapped[i, 1])
            j = int(np.argmin(d))
            if d[j] <= 25:
                snapped[i] = tgt[j]
        allNode = snapped

    _stage(f'Reeb2+combine done: No_node={allNode.shape[0]} cells={len(splitReg_work)} GCC={GCC}')
    if _live:
        _live.update("Reeb + combined graph", cells=list(splitReg_work), critP=subcritP, reeb=reeb)
    LOG.kv("Reeb", f"edges: {reebEdge.shape[0] if reebEdge.size else 0}   "
           f"cells to sweep: {len(splitReg_work)}")
    LOG.kv("combined graph", f"{allNode.shape[0]} nodes, "
           f"{(crackEdge.shape[0] if crackEdge.size else 0) + (reebEdge.shape[0] if reebEdge.size and not GCC else 0)} "
           f"required edges" + ("   (GCC: crack-only)" if GCC else ""))

    # ---- optional RPP pre-augmentation: connect components + match odd-degree nodes ----
    aug_edges = []
    if _RPP_RECONNECT and not GCC and np.size(allEdge):
        aug_edges = _rpp_augment(allEdge, allNode)
        if aug_edges:
            allEdge = np.vstack([allEdge, np.array(aug_edges, int)])
            LOG.kv("RPP reconnect", f"{len(aug_edges)} optimal links (Frederickson) "
                   f"-> graph connected + Eulerian")

    No_node = allNode.shape[0]
    allEdge = np.vstack([allEdge, allEdge[:, ::-1]])
    adj = np.zeros((No_node, No_node))
    for (u, v) in allEdge:
        adj[u, v] += 1          # COUNT parallel Reeb edges (accumulate, not clobber). Two cells split by
        #                         a crack share a crit-point pair -> parallel edge; `=1` collapsed
        #                         them so the CPP swept only one of the two cells. (Needs the
        #                         ChinesePostman multigraph fix in _fleury_row case-2.)
    Dist = squareform(pdist(allNode))
    # Reeb-edge weight = cell area / (2*(s/sqrt2)): a sweep-length proxy that does not depend
    # on exact critical-point locations. Guarded by ~GCC (the GCC branch drops subcritP from
    # allNode, so offset Reeb indices would exceed Dist).
    if not GCC and np.size(reebEdge_off):
        sensor = s / math.sqrt(2)
        reb = np.atleast_2d(np.asarray(reebEdge_off, int))
        for rb in range(reb.shape[0]):
            if rb < len(splitReg_work):
                g = splitReg_work[rb]
                g = g.geom if hasattr(g, "geom") else g
                w = float(g.area) / (2 * sensor)
                u, v = int(reb[rb, 0]), int(reb[rb, 1])
                Dist[u, v] = w
                Dist[v, u] = w
    AdjMax = adj * Dist

    _stage('calling ChinesePostman')
    LOG.sub("ChinesePostman route")
    Path, weight, add, st = ChinesePostman(adj, AdjMax, Dist, [], [])
    if aug_edges and _RPP_MODE == "full":            # Eulerian pre-augmented -> reconnect set is ours
        add = np.array(aug_edges, int)
    _stage(f'CPP done: path_nodes={np.atleast_1d(Path).size}')
    if _live:
        _ai = np.atleast_2d(add).astype(int) if np.size(add) else np.empty((0, 2), int)
        _aN = np.atleast_2d(allNode)
        _adds = (np.stack([_aN[_ai[:, 0]][:, ::-1], _aN[_ai[:, 1]][:, ::-1]], axis=1)
                 if _ai.size else np.empty((0, 2, 2)))
        _live.update("ChinesePostman route", adds=_adds)
    LOG.kv("route", f"{np.atleast_1d(Path).size} nodes   "
           f"reconnect links: {np.atleast_2d(add).shape[0] if np.size(add) else 0}"
           + ("   (Frederickson RPP)" if aug_edges else ""))
    Path = np.atleast_1d(np.asarray(Path, int))
    Path_pairs = np.column_stack([Path[:-1], Path[1:]])

    if GCC:
        # GCC branch (not exercised at k=8): drop the longest matched edge
        add_arr = np.atleast_2d(np.asarray(add, int)) if np.size(add) else np.empty((0, 2), int)
        from ImagePlanning_oSCC import _ismember_rows
        is_add = _ismember_rows(Path_pairs, add_arr) | _ismember_rows(Path_pairs[:, ::-1], add_arr)
        ppPath = Path_pairs.copy()
        ppPath[~is_add] = 0
        m = int(np.argmax(spdist2(allNode[ppPath[:, 0]], allNode[ppPath[:, 1]])))
        Path_pairs = np.delete(Path_pairs, m, axis=0)
        idx = np.concatenate([Path_pairs[:, 0], [Path_pairs[-1, 1]]])
        PathEdge = allNode[idx][:, ::-1]
    else:
        wall_fol = np.zeros(Path_pairs.shape[0])
        subXY = Boustrophedon_CellCon(Path_pairs, list(splitReg_work), [], [], critP, wall_fol,
                                      True, False, reebEdge_off, reebCell, allNode, s, a,
                                      crack_reach=crack_reach, n_crack=np.atleast_2d(node).shape[0])
        subXY = np.atleast_2d(subXY)
        _subXY_raw = subXY.copy()                    # pre-cut capture
        _stage(f'Boustrophedon done: poses={subXY.shape[0]}')
        LOG.sub("Boustrophedon coverage + crack fill")
        LOG.kv("path poses", subXY.shape[0])
        # start/end optimization: cut the closed Euler tour at its LONGEST hop so the
        # open path's endpoints are that hop's vertices. With 0-based argmax the
        # equivalent roll is N-m-1 (not N-m, which would land start==end).
        subXY[-1] = subXY[0]
        m = int(np.argmax(spdist2(subXY[:-1], subXY[1:])))
        subXY = subXY[:-1]
        subXY = np.roll(subXY, subXY.shape[0] - m - 1, axis=0)
        PathEdge = np.vstack([PathEdge, subXY]) if PathEdge.size else subXY
        if _live:
            _live.update("coverage path", path=PathEdge, clear=("reeb",))

    # ---- results ----
    pathLength = total_length(PathEdge)
    scclen = pxinMap(pathLength) / 12
    areaCover = 2 * pxinMap(s / math.sqrt(2)) / 12 * scclen
    areaCover1 = (pu.polybuffer(PathEdge, s / math.sqrt(2), kind="lines").area / (2898 * 3050)) * 100
    res = [areaCover1, _DEN[dd - 1], ttt, scclen, areaCover]

    if LOG.enabled:
        LOG.section("RESULT")
        LOG.kv("map density", f"{int(_DEN[dd - 1])}%")
        LOG.kv("sensor coverage", f"{areaCover1:.2f}%   (of the 3050x2898 workspace)")
        LOG.kv("path length", f"{scclen:.3f} ft")
        LOG.kv("area covered", f"{areaCover:.3f} sq ft")
        LOG.kv("path poses", int(np.atleast_2d(PathEdge).shape[0]))
        if LOG.path:
            LOG.kv("log saved", LOG.path)

    if _live is not None:
        _live.update("coverage path", path=PathEdge, clear=("reeb",))   # final (also covers GCC)
        _live.animate_robot(PathEdge, s, a)          # robot sweeps the full path
        _live.hold()
    if viz is not None or out is not None:           # ALWAYS write the final-path PNG (+ GIF for save)
        _scc_viz(viz, img_n, out, res, cells=list(splitReg_work), subcritP=subcritP,
                 reeb=reeb, node=node, crackEdge=crackEdge, add=add, allNode=allNode,
                 PathEdge=PathEdge, skel=skel, a=a)
    if _out is not None:                             # non-invasive capture (sweep scoring)
        _out["Path"] = [int(x) for x in np.atleast_1d(Path).ravel()]
        _out["n_node"] = int(np.atleast_2d(node).shape[0])      # crack-graph size
        _out["n_edge"] = int(np.atleast_2d(edgeList).shape[0])
        _out["WSarea"] = WSarea
        _out["coverPercent"] = (areaCover / WSarea) if WSarea else float("nan")
        _out["allNode"] = np.asarray(allNode, float)
        if not GCC:
            _out["subXY_raw"] = _subXY_raw
            _out["cut_m"] = int(m)
            # celled-rpp inputs for the best-of-3 3rd candidate (cells are order-invariant).
            _out["_rpp"] = dict(
                node=np.asarray(node, float), crackEdge=np.asarray(edgeList, int),
                cell_geoms=[c.geom for c in splitReg_work],
                cell_verts=[np.atleast_2d(np.asarray(c.Vertices, float)) for c in splitReg_work],
                s=float(s), a=float(a))
    return PathEdge, res


def _scc_events(img_n, *, cells, subcritP, reeb, node, crackEdge, add, allNode, PathEdge, skel, a):
    """Build the OnlineSCC-compatible event stream from SCC state (event-pipeline
    approach): init + one decomposition (with SCC-specific overlays) + a pose per
    final-path point. SCC overlays ride in the iter event under ``scc_*`` keys."""
    if skel is None:                                  # crack skeleton for the figure
        from ImagePlanning_SCC import _front_end
        skel = _front_end(img_n, a)
    sk = np.argwhere(np.asarray(skel) > 0)            # (row,col)
    rowBW, colBW = np.asarray(skel).shape
    PE = np.atleast_2d(PathEdge)[:, :2]

    nfx = np.atleast_2d(node)[:, ::-1] if np.size(node) else np.empty((0, 2))   # crack nodes (x,y)
    aN = np.atleast_2d(allNode)
    addi = np.atleast_2d(add).astype(int) if np.size(add) else np.empty((0, 2), int)
    adds_xy = (np.stack([aN[addi[:, 0]][:, ::-1], aN[addi[:, 1]][:, ::-1]], axis=1)
               if addi.size else np.empty((0, 2, 2)))     # (K,2,2) endpoint pairs in (x,y)
    events = [
        dict(kind="init", skeleton=sk, rowBW=rowBW, colBW=colBW, planner="SCC"),
        dict(kind="iter", it=1,
             cells=[np.atleast_2d(c.boundary())[:, :2] for c in cells],
             critP=(np.atleast_2d(subcritP)[:, ::-1] if np.size(subcritP) else np.empty((0, 2))),
             reeb=[np.atleast_2d(np.asarray(e, float))[:, [1, 0]] for e in (reeb or [])],
             reebEdge=np.empty((0, 2), int), reebCell=np.empty(0, int), Path=[], wall_fol=[],
             curPt=(float(PE[0, 0]), float(PE[0, 1])), raw_subXY=PE,
             scc_crackNodes=nfx,
             scc_crackEdges=np.atleast_2d(crackEdge) if np.size(crackEdge) else np.empty((0, 2), int),
             scc_adds=adds_xy),
    ]
    for i, p in enumerate(PE):
        events.append(dict(kind="pose", it=1, i=i, curPt=(float(p[0]), float(p[1]))))
    return events


def _scc_viz(viz, img_n, out, res, **state):
    """Render SCC state via OnlineSCC's render_animation (through VizOptions). The single
    final-path PNG (written in every mode) goes to Results/SCC/; GIFs (save modes) go to
    Results/GIF/. ``out`` overrides the matching output by extension (.png -> the PNG,
    .gif -> the GIF)."""
    from private.visualize import VizOptions
    opt = viz if isinstance(viz, VizOptions) else VizOptions(mode=viz, style="color")
    tag = img_n if isinstance(img_n, str) else "crackGen"
    opt.title = f"SCC — {tag}"
    opt.final_png = out if (out and out.endswith(".png")) else f"Results/SCC/{tag}_SCC.png"
    opt.out_gif = out if (out and out.endswith(".gif")) else f"Results/GIF/{tag}_SCC.gif"
    opt.events = _scc_events(img_n, **state)
    return opt.finalize()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Offline SCC planner (known crack map).",
        epilog="Robot dimensions (base / footprint / sensor diameters) are configured "
               "in robot_config.json -- edit that file to change the robot.")
    p.add_argument("map", nargs="?", default=None,
                   help="explicit crack-map name (e.g. myCrack8_100_1). Overrides --den/--sig/--map-num; "
                        "omit it to build the map from those flags.")
    p.add_argument("--den", type=int, default=100, choices=_DEN, metavar="PCT",
                   help="crack density %% (one of 35/45/50/65/80/90/95/100; default 100)")
    p.add_argument("--sig", type=int, default=None, choices=[5, 10, 20], metavar="S",
                   help="Gaussian sigma (5/10/20). If given -> a GAUSSIAN map; omitted -> UNIFORM.")
    p.add_argument("--map-num", dest="mapnum", type=int, default=None, metavar="N",
                   help="map number: Uniform -> variant 1-5 (random if omitted); "
                        "Gaussian -> set folder 1-6 (random if omitted).")
    p.add_argument("--viz", default=None, choices=["show", "save", "show+save", "live"],
                   help="output mode: show=open viewer; save=write GIF+PNGs; show+save=both; "
                        "live=interactive per-stage debug figure that redraws at each pipeline "
                        "stage (MCD/Reeb/route/path) -- set a breakpoint to inspect each stage. "
                        "The single final-path PNG is ALWAYS written to Results/SCC/ (every mode); "
                        "GIFs go to Results/GIF/. OMIT --viz: just the final-path PNG.")
    p.add_argument("--style", default="color", choices=["plain", "publish", "color"],
                   help="render richness")
    p.add_argument("--step", type=int, default=1,
                   help="poses per coverage frame (save modes; larger -> shorter GIF)")
    p.add_argument("--fps", type=int, default=14, help="GIF frames per second (save modes)")
    p.add_argument("--smooth", type=float, default=0.0, metavar="M",
                   help="tween the robot between poses at this spacing in metres (save modes; 0 = off)")
    p.add_argument("--gif-max-px", dest="gif_max_px", type=int, default=1100,
                   help="cap the GIF's larger side in pixels (save modes)")
    p.add_argument("--frames", default=None, metavar="DIR",
                   help="also write per-step PNG frames to DIR (save modes)")
    p.add_argument("--route", default=None, choices=["rpp", "best", "native", "matlab"],
                   help="coverage-route formulation (default: env OSCC_SCC_ROUTE, else 'rpp').  "
                        "rpp = geometry-routed celled-rpp, a single deterministic order-invariant pass "
                        "(cheapest single formulation, 100%% sensor coverage + 100%% crack fill on all maps).  "
                        "best = coverage-guarded best-of-3 (native + boundary-start + rpp).  "
                        "native = single reeb-graph ChinesePostman with the native MCD critical order.  "
                        "matlab = single reeb-graph ChinesePostman with the boundary-scan critical order.")
    p.add_argument("-o", "--out", default=None, help="output path (GIF for save modes; PNG for default)")
    p.add_argument("-q", "--quiet", action="store_true", help="silence console logging (file still saved)")
    args = p.parse_args()

    from private.mapselect import resolve_map
    img_n, dd, desc = resolve_map(args.map, den=args.den, sig=args.sig, mapnum=args.mapnum)

    # logging ON by default: console (unless -q) + a rotating file logs/SCC_<map>.log
    LOG.enable(console=not args.quiet, logfile=rotate_path("logs", f"SCC_{img_n.replace('/', '_')}"))
    print(f"map: {desc}")

    from private.visualize import VizOptions
    _v = VizOptions(mode=args.viz or "final", style=args.style,   # no --viz -> Final-Path PNG
                    step=args.step, fps=args.fps, smooth_step_m=args.smooth,
                    gif_max_px=args.gif_max_px, frame_dir=args.frames)
    PathEdge, res = run_scc(img_n, dd=dd, viz=_v, out=args.out, route=args.route)
    print("res =", [round(float(x), 3) for x in res])

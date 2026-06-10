"""Online Sensor-based Complete Coverage (OnlineSCC) planner.

OnlineSCC plans for the same crack-filling robot as SCC, but for an UNKNOWN work
area: crack locations are not given. The robot scans the area in a boustrophedon
(zig-zag) pattern and, whenever its sensor detects a crack within range, plans
and fills that crack before resuming the scan -- repeating until the whole area
is covered.

Outer coverage loop (repeats until the area is fully covered):
  1. Decompose the not-yet-covered free space at its critical points into monotone
     coverage cells (Morse Cell Decomposition, MCD -> Reeb), order them, and lay a
     boustrophedon (zig-zag) sweep through them.
  2. Inner sensing/fill loop -- walk that sweep pose by pose:
       a. sense the cracks within sensor range and accumulate them;
       b. when an accumulated crack is fillable, plan its fill waypoints
          (ImagePlanning_oSCC) and splice them into the sweep.
  3. Subtract the swept region from the free space and re-decompose the rest.

Robot model: the base / footprint / sensor diameters (configured in
``robot_config.json``) give the base radius r1, the fill-footprint radius a and
the sensor radius s; the zig-zag spacing is s/sqrt2.

Entry point: ``run_online_scc(...) -> (PathEdge, res)``. Run ``python OnlineSCC.py``
to select a map and plan it; see ``--help``.
"""

import math
import os as _os

import numpy as np

from private import poly_utils as pu
from private.utils import (argwhere2d, bound, inpxMap, mfix, mround, pxinMap, spdist,
                   spdist2, total_length)
from MCD import MCD
from Reeb import Reeb
from ReebPath import reeb_traversal
from Boustrophedon import Boustrophedon
from Boustrophedon_CellCon import Boustrophedon_CellCon
import Boustrophedon_CellCon as _BCC
from ImagePlanning_oSCC import ImagePlanning_oSCC
from private.endP_ident import endP_ident
from private.compCrack import compCrack
from private.bwmorph import neighbor_count_points, _neighbor_counts_at
from private.DecimatePoly import DecimatePoly
from private.run_log import LOG, rotate_path
from private.config_loader import ROBOT1

_DIR_MAP = np.array([[-1, -1], [-1, 0], [-1, 1],
                     [0, -1], [0, 1],
                     [1, -1], [1, 0], [1, 1]])
_DEN = [35, 45, 50, 65, 80, 90, 95, 100]


# --------------------------------------------------------------------------- #
# polygon helpers
# --------------------------------------------------------------------------- #
def _reg_list(ps):
    """regions() as a Python list of PolyShape (single region -> [ps])."""
    if isinstance(ps, (list, tuple)):
        out = []
        for p in ps:
            out.extend(_reg_list(p))
        return out
    rs = pu.regions(ps)
    return list(rs)


def _regJoin(polyin):
    """Join a list of single-region PolyShapes into one multi-boundary PolyShape.

    Each region is sanitized first (non-finite vertices dropped) before the union;
    this prevents NaN/Inf from propagating through later polygon operations.
    """
    geoms = []
    for p in polyin:
        g = pu._as_geom(pu.sanitize(p))
        if g is not None and not g.is_empty:
            geoms.append(g)
    if not geoms:
        return pu.polyshape()
    from shapely.ops import unary_union
    # union the region GEOMETRIES (hole-correct) rather than feeding NaN-separated
    # .Vertices to a boundary-constructor, which would build an invalid Polygon for
    # any region with holes/multiple loops (NaN in orientationIndex).
    return pu.PolyShape(unary_union(geoms))


def _polyclean(polyin):
    """Drop regions with area*1e-3 <= 5 (integer-truncated) and rejoin the rest."""
    if isinstance(polyin, (list, tuple)):
        poly = [p for p in polyin if mfix(p.area * 1e-3) > 5]
        return poly
    poly = [r for r in _reg_list(polyin) if mfix(r.area * 1e-3) > 5]
    return _regJoin(poly)


def _regCombine(polyin, r=20.0):
    """Morphological-open union: buffer each region by r, union, then shrink by r."""
    geoms = []
    polyin = polyin if isinstance(polyin, (list, tuple)) else _reg_list(polyin)
    for p in polyin:
        g = pu._as_geom(p)
        if g is not None and not g.is_empty:
            geoms.append(g.buffer(r, quad_segs=pu._QSEG))
    if not geoms:
        return pu.polyshape()
    from shapely.ops import unary_union
    u = unary_union(geoms).buffer(-r, quad_segs=pu._QSEG)
    return pu.PolyShape(u)


def _regions_area(ps):
    return np.array([r.area for r in _reg_list(ps)], dtype=float)


def _sMaskid(BW, rr, cp):
    """Return the (row_idx, col_idx) bounding-box slice (0-based) of a radius-rr
    disk centred at cp=(x,y), clamped to BW's dimensions."""
    H, W = BW.shape
    r0 = int(bound(mround(cp[1] - rr), 1, H))
    r1 = int(bound(mround(cp[1] + rr), 1, H))
    c0 = int(bound(mround(cp[0] - rr), 1, W))
    c1 = int(bound(mround(cp[0] + rr), 1, W))
    rows = np.arange(r0 - 1, r1)        # inclusive range, 0-based
    cols = np.arange(c0 - 1, c1)
    return rows, cols


def _ismember_rows(A, B):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.size == 0:
        return np.zeros(0, dtype=bool)
    if B.size == 0:
        return np.zeros(A.shape[0], dtype=bool)
    # Vectorized over A's rows. np.isclose is asymmetric (rtol scales the 2nd arg),
    # so B is kept first to preserve the same near-tolerance semantics.
    return np.isclose(B[None, :, :], A[:, None, :]).all(axis=2).any(axis=1)


# --------------------------------------------------------------------------- #
# image front-end
# --------------------------------------------------------------------------- #
def _bwmorph_fill(BW):
    """Fill isolated single-pixel background holes: set a 0 pixel to 1 when
    all four 4-connected neighbours are 1."""
    BW = np.asarray(BW) > 0
    up = np.zeros_like(BW); up[1:, :] = BW[:-1, :]
    dn = np.zeros_like(BW); dn[:-1, :] = BW[1:, :]
    lf = np.zeros_like(BW); lf[:, 1:] = BW[:, :-1]
    rt = np.zeros_like(BW); rt[:, :-1] = BW[:, 1:]
    hole = (~BW) & up & dn & lf & rt
    return BW | hole


def _bwmorph_spur(BW, n):
    """Prune skeleton spurs by removing endpoint pixels n times.

    Uses a more aggressive endpoint-removal strategy than the conservative
    LUT-based spur algorithm, which is appropriate for the denser spur set
    produced by the skimage Lee thinning skeleton."""
    from private.bwmorph import neighbor_count_points
    BW = (np.asarray(BW) > 0).astype(int)
    for _ in range(int(n)):
        eP, _ = neighbor_count_points(BW, 1)        # 1 = endpoints
        if not eP.any():
            break
        BW[eP > 0] = 0
    return BW


def _preprocess(img_n):
    """Load a Uniform PNG crack map, binarize it (Otsu threshold + area filter),
    complement, pad, fill single-pixel holes, and skeletonize.

    Returns the skeleton as a 0/1 int array (same shape as the padded image).
    """
    from skimage.io import imread
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from private.bwskel import bwskel

    img = imread(f"CrackMaps/Uniform/{img_n}.png")
    ch = img[:, :, 0] if img.ndim == 3 else img
    BW = ch > threshold_otsu(ch)
    # bwareaopen(50) analogue. NOTE: skimage 0.26 deprecated `min_size`; `max_size=51`
    # max_size=51 removes components of size <=51 (same as the deprecated min_size=51)
    # and silences the skimage FutureWarning.
    BW = remove_small_objects(BW, max_size=51, connectivity=2)
    BW = ~BW
    BW = np.pad(BW, 1)
    BW = _bwmorph_fill(BW)                       # fill isolated single-pixel holes
    BW2 = bwskel(BW)                             # Lee-94 3D thinning (private.bwskel)
    BW2 = _bwmorph_spur(BW2, 10)                 # prune short spurs 10 times
    return BW2.astype(int)


def _preprocess_by_name(img_n):
    """Front-end dispatch by map name: a Gaussian name ('Gaussian<b>/myCrackGauss_...'
    or a bare 'myCrackGauss_...') loads the .mat crackGen mask -> _preprocess_crackgen;
    anything else is a Uniform PNG -> _preprocess."""
    if isinstance(img_n, str) and "Gauss" in img_n:
        import os
        from scipy.io import loadmat
        if os.sep in img_n or "/" in img_n:
            matpath = os.path.join("CrackMaps", img_n.replace("/", os.sep) + ".mat")
        else:
            matpath = next((p for b in range(1, 9)
                            for p in [os.path.join("CrackMaps", f"Gaussian{b}", img_n + ".mat")]
                            if os.path.exists(p)), None)
            if matpath is None:
                raise FileNotFoundError(f"Gaussian map '{img_n}' not found in CrackMaps/Gaussian1..8")
        return _preprocess_crackgen(np.asarray(loadmat(matpath)["crackGen"]))
    return _preprocess(img_n)


def _preprocess_crackgen(crackGen):
    """Ingest a preloaded Gaussian crack map (crackGen array from a .mat file).

    Applies fill -> skeletonize -> spur-prune without the binarize/pad steps
    that are only needed for the Uniform PNG front-end. Also zeroes the last
    pixel of the array (artifact of the source data format).
    """
    from private.bwskel import bwskel
    BW = (np.asarray(crackGen) > 0)
    BW = BW.copy(); BW[-1, -1] = False           # zero the last pixel (source-data artifact)
    BW = _bwmorph_fill(BW)                        # fill isolated single-pixel holes
    BW2 = bwskel(BW)                             # Lee-94 3D thinning (private.bwskel)
    BW2 = _bwmorph_spur(BW2, 10)                 # prune short spurs 10 times
    return BW2.astype(int)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _viz_emit(_viz, event):
    """Append a viz event to a plain list (capture-only) or a VizOptions (capture)."""
    if _viz is None:
        return
    (_viz.capture if hasattr(_viz, "capture") else _viz.append)(event)


def run_online_scc(img_n="myCrack8_100_1",
                   bw_working=None, dd=8, max_iter=None, _trace=None, _max_pose=None,
                   _inject=None, _iter_trace=None, _force=None, _viz=None, _out=None):
    """Run the online scan-and-fill plan for one crack map.

    Parameters
      img_n    : crack-map name under CrackMaps/ -- a Uniform map, or a Gaussian
                 'Gaussian<b>/myCrackGauss_s<sig>_<den>' name.
      dd       : crack-density index (1..8); used only to label the result row.
      max_iter : optional cap on the number of outer coverage iterations.
      _viz     : "show"/"save"/"show+save"/"live" to display or write per-step frames
                 of the scan and fills (assembled into an animation), or a list to
                 capture the step events. The final-path PNG goes to Results/OnlineSCC/.

    Returns
      PathEdge : (M, 2) full scan+fill path as (x, y) points.
      res      : [iterations, density %, run-time s, path length ft, area covered sq ft].
    """
    # _viz events (separable from the planner): {"kind":"iter", ...} per decomposition,
    # {"kind":"pose", ...} per inner-loop pose.
    if isinstance(_viz, str):                        # shorthand: _viz="pos"/"show"/...
        from private.visualize import VizOptions             # lazy: keep the planner viz-free
        _viz = VizOptions(mode=_viz)
    # robot dimensions from robot_config.json (edit there to reconfigure the robot)
    botD = ROBOT1.base_diameter_in
    footD = ROBOT1.footprint_diameter_in
    sensD = ROBOT1.sensor_diameter_in
    r1, a, s = ROBOT1.r1, ROBOT1.a, ROBOT1.s   # base / footprint / sensor radius (px)
    sm_flag = 1

    # --- logging (no-op unless enabled; CLI enables console+file, env enables console) ---
    import time as _time
    if not LOG.enabled and _os.environ.get("OSCC_DRV_DBG"):
        LOG.enable(console=True)
    _t_run = _time.time()
    LOG.section(f"OnlineSCC  |  map: {img_n}")
    LOG.kv("algorithm", "OnlineSCC (online, environment discovered while scanning)")
    LOG.kv("map", img_n)
    LOG.kv("robot base", f"{botD}in dia  ->  r1 = {r1} px")
    LOG.kv("footprint", f"{footD}in dia  ->  a = {a} px")
    LOG.kv("sensor", f"{sensD:.0f}in dia  ->  s = {s} px   (zig-zag spacing {s / math.sqrt(2):.1f} px)")

    if bw_working is not None:
        BW_working = (np.asarray(bw_working) > 0).astype(int)
    else:
        BW_working = _preprocess_by_name(img_n)

    BW3 = BW_working.copy()
    rowBW, colBW = BW3.shape
    if _viz is not None:                             # full ground-truth crack skeleton
        _sk = np.argwhere(BW_working > 0)            # (row,col) -- thin underlay for viz
        _viz_emit(_viz, dict(kind="init", skeleton=_sk, rowBW=rowBW, colBW=colBW))
    crackGen = np.zeros_like(BW_working)
    crackGGen = np.zeros_like(BW_working)
    m_all, n_all = np.nonzero(BW_working)
    mBW = list(m_all)
    nBW = list(n_all)

    Y = [0, colBW, colBW, 0]
    X = [0, 0, rowBW, rowBW]
    extBound = pu.polyshape(Y, X)
    final = extBound
    final_work = extBound
    critP = 0                                    # scalar 0 on iteration 1 (no prior path end)
    loop1 = True
    aa = pu.polybuffer([0, 0], a, kind="points")
    PathEdge = np.array([[0.0, 0.0]])
    numItr = 0
    iMor = 0
    WSarea = 0.0

    # Debug: inject pre-computed geometry for one iteration (final/final_work
    # free-space boundaries, critP, PathEdge end-point) to validate the coverage
    # path in isolation from the cumulative subtract drift.
    if _force is not None:
        fa = np.atleast_2d(np.asarray(_force["final"], float))
        fwb = np.atleast_2d(np.asarray(_force["final_work"], float))
        final = pu.polyshape(fa[:, 0], fa[:, 1])
        final_work = pu.polyshape(fwb[:, 0], fwb[:, 1])
        critP = _force["critP"]
        PathEdge = np.atleast_2d(np.asarray(_force["pathend"], float))
        # Inject the accumulated crack/BW state entering this iteration so the
        # inner sensing/fill loop starts where the previous iteration left off.
        bwst = _force.get("bw")
        if bwst is not None:
            BW_working = np.asarray(bwst["BW_working"], dtype=int)
            BW3 = np.asarray(bwst["BW3"], dtype=int)
            crackGen = np.asarray(bwst["crackGen"], dtype=int)
            crackGGen = np.asarray(bwst["crackGGen"], dtype=int)
            mBW = list(np.asarray(bwst["mBW"], dtype=int).ravel())
            nBW = list(np.asarray(bwst["nBW"], dtype=int).ravel())
            numItr = int(bwst.get("numItr", 0))

    it = 0
    _iters = 0
    _iter_stats = []                                  # (it, scan poses, total poses, crack-detect) for the table
    while np.any(_regions_area(final) > 2 * aa.area):
        it += 1
        if max_iter is not None and it > max_iter:
            break
        _iters = it
        LOG.sub(f"Main loop iteration {it}   (numItr so far {numItr}, t={_time.time() - _t_run:.1f}s)")

        # per-iteration outer-loop capture (debug): free space ENTERING this iter
        if _iter_trace is not None:
            try:
                _fw_in = np.atleast_2d(final_work.boundary())[:, :2].copy()
            except Exception:
                _fw_in = np.empty((0, 2))

        # ---- MCD prep ----
        # Morphological open (erode s/5, dilate s/5) to drop sub-footprint slivers.
        # The open can over-erode surviving regions, so the result is used only if
        # it preserves the area and region count; otherwise final_work is used directly.
        import os as _os2
        working = pu.polybuffer(pu.polybuffer(final_work, -s / 5), s / 5)
        # The GEOS buffer erode can over-remove area from surviving regions, shifting
        # MCD critical-point locations. The default behaviour preserves the un-eroded
        # shapes of surviving regions; OSCC_RAW_PREP_OPEN=1 restores the raw open.
        if not _os2.environ.get("OSCC_RAW_PREP_OPEN"):
            fa = final_work.area
            if fa > 0 and (fa - working.area) / fa > 0.005:
                # region-preserving case -> working == final_work; sliver-removal
                # case (e.g. iter-6: 4 regions -> 3) keeps the survivors un-eroded.
                keep = [r for r in _reg_list(final_work)
                        if pu.polybuffer(r, -s / 5).area != 0]
                working = _regJoin(keep) if keep else final_work
        wregs = [r for r in _reg_list(working) if r.area > aa.area]
        working = wregs
        fw_regs = _reg_list(final_work)
        if len(fw_regs) > 1:
            fw_regs = [r for r in fw_regs if pu.polybuffer(r, -s / 5).area != 0]
            final_work = _regJoin(fw_regs)
        if len(working) != len(_reg_list(final_work)):
            working = final_work
        else:
            working = _regJoin(working) if isinstance(working, list) else working

        # Debug: inject a pre-computed post-prep working boundary to isolate
        # a coverage divergence to prep vs MCD-onward.
        if _force is not None and _force.get("working") is not None:
            wb = np.atleast_2d(np.asarray(_force["working"], float))
            working = pu.polyshape(wb[:, 0], wb[:, 1])

        subcritP, splitReg_work, splitReg, splitEdge = MCD(working, final_work, final, critP, sm_flag)
        splitReg_work = _polyclean(splitReg_work)
        if not isinstance(splitReg_work, list):
            splitReg_work = _reg_list(splitReg_work)
        subcritP = np.atleast_2d(subcritP)[:, ::-1]      # MCD returns (x,y); convert to (row,col)
        iMor += 1
        if subcritP.size == 0:
            LOG.kv("MCD", "no critical points -> coverage complete")
            break
        if LOG.enabled:
            LOG.kv("MCD", f"critical points: {subcritP.shape[0]}   cells: {len(splitReg_work)}"
                          f"   split-edges: {len(np.atleast_2d(splitEdge)) if np.size(splitEdge) else 0}")

        # ---- Reeb (first pass) + prune stray critical points ----
        reebEdge, _, _, _, remreg = Reeb(splitReg_work, subcritP, splitEdge)
        if len(remreg):
            splitReg_work = [c for k, c in enumerate(splitReg_work) if k not in set(np.atleast_1d(remreg).astype(int))]
        reebEdge = np.atleast_2d(np.asarray(reebEdge, dtype=int))

        subcritP, splitReg_work, reebEdge = _prune_stray(
            subcritP, splitReg_work, reebEdge, s)

        reebEdge, reebCell, reeb, reebwall, remreg = Reeb(splitReg_work, subcritP, splitEdge)
        if len(remreg):
            splitReg_work = [c for k, c in enumerate(splitReg_work) if k not in set(np.atleast_1d(remreg).astype(int))]
        reebEdge = np.atleast_2d(np.asarray(reebEdge, dtype=int))
        reebCell = np.asarray(reebCell, dtype=int)
        if LOG.enabled:
            LOG.kv("Reeb", f"edges: {reebEdge.shape[0] if reebEdge.size else 0}"
                           f"   cells: {reebCell.size}   nodes: {subcritP.shape[0]}")

        allNode = subcritP
        # simple 0/1 adjacency
        N = subcritP.shape[0]
        adj = np.zeros((N, N))
        for u, v in reebEdge:
            adj[u, v] = 1
            adj[v, u] = 1

        se, seP = [], []

        # ---- coverage sweep ----
        if len(splitReg_work) == 1:
            ind = _start_node(reebEdge, reebCell, splitReg_work, subcritP, PathEdge)
            Path, wall_fol, _ = reeb_traversal(adj.copy(), subcritP, reebEdge, splitReg_work, ind)
            subXY = Boustrophedon(Path, splitReg_work, se, seP, critP, wall_fol, False, False,
                                  reebEdge, reebCell, allNode, s, a)
        else:
            # Global-shortest-path start selection: when multiple cells are within
            # eps of the nearest, try each candidate's reeb_traversal order and
            # keep the one with the smallest Boustrophedon_CellCon path length.
            cands = _start_candidates(reebEdge, reebCell, splitReg_work, subcritP, PathEdge)
            best, costs = None, []
            for cand in cands:
                P, wf, _ = reeb_traversal(adj.copy(), subcritP, reebEdge, splitReg_work, cand)
                # fresh see/seP per candidate; a shared list would leak across calls
                sub = Boustrophedon_CellCon(P, list(splitReg_work), [], [], critP, wf, False, False,
                                            reebEdge, reebCell, allNode, s, a)
                cost = float(min(_BCC._LAST.get("dists") or [float("inf")]))
                costs.append((int(cand), round(cost, 1)))
                if best is None or cost < best["cost"]:
                    best = dict(ind=int(cand), Path=P, wall_fol=wf, subXY=sub, cost=cost,
                                ccPath=list(_BCC._LAST.get("ccPath", [])))
            if _os.environ.get("OSCC_SN_DBG"):
                print(f"[start-sel] it={it} ncells={len(splitReg_work)} cands/costs={costs} "
                      f"-> ind={best['ind']}", flush=True)
            ind, Path, wall_fol, subXY = best["ind"], best["Path"], best["wall_fol"], best["subXY"]
            _sel_ccPath = best.get("ccPath", [])
        subXY = np.atleast_2d(subXY)
        if LOG.enabled:
            _cc = locals().get("_sel_ccPath") or list(_BCC._LAST.get("ccPath", []))
            LOG.kv("ReebPath", f"cells swept: {len(splitReg_work)}   "
                               f"traversal: {[int(x) for x in np.ravel(Path)]}")
            if _cc:
                LOG.kv("cell order", f"ccPath: {[int(x) for x in _cc]}")
            LOG.kv("Boustrophedon", f"{subXY.shape[0]} scan poses")

        if _viz is not None:
            cells_xy = [np.atleast_2d(c.boundary())[:, :2] for c in splitReg_work]
            critP_xy = (np.atleast_2d(subcritP)[:, ::-1].copy()    # (row,col)->(x,y)
                        if np.size(subcritP) else np.empty((0, 2)))
            reeb_xy = [np.atleast_2d(np.asarray(e, float))[:, [1, 0]] for e in (reeb or [])]
            #          ^ reeb (+ reebwall) control points are (row,col); flip to (x,y) for viz
            reebwall_xy = [np.atleast_2d(np.asarray(e, float))[:, [1, 0]] for e in (reebwall or [])]
            # critP_xy doubles as node coords (subcritP, x/y) -> indexable by Path node id.
            # curPt = PathEdge end ENTERING this iter (robot pos for the Reeb connecting edge).
            _viz_emit(_viz, dict(kind="iter", it=it, cells=cells_xy, critP=critP_xy,
                                 reeb=reeb_xy, reebEdge=np.atleast_2d(reebEdge).copy(),
                                 reebwall=reebwall_xy,
                                 reebCell=np.asarray(reebCell).copy(),
                                 wall_fol=[int(w) for w in (wall_fol or [])],
                                 Path=[[int(x) for x in row] for row in (Path or [])],
                                 curPt=np.asarray(PathEdge, float)[-1, :2].copy(),
                                 raw_subXY=subXY[:, :2].copy()))

        if _iter_trace is not None:
            _it_reebEdge = np.atleast_2d(reebEdge).copy()
            _it_path = [list(np.atleast_1d(p)) for p in Path] if len(Path) else []
            _it_ind = int(ind)
            _it_ccPath = _sel_ccPath if len(splitReg_work) > 1 else []   # winning ccPath
            _it_raw_subXY = subXY[:, :2].copy()       # Boustrophedon sweep, pre-fill

        if loop1:
            ziglen = pxinMap(total_length(subXY)) / 12
            WSarea = 2 * pxinMap(s / math.sqrt(2)) / 12 * ziglen

        subXY = np.column_stack([subXY, np.zeros(subXY.shape[0])])    # 3rd col = fill flag

        # ---- inner sensing/fill loop ----
        state = dict(crackGen=crackGen, crackGGen=crackGGen, BW3=BW3, BW_working=BW_working,
                     mBW=mBW, nBW=nBW, PathEdge=PathEdge, numItr=numItr,
                     _trace=(_trace if (it == 1 or _os.environ.get("OSCC_TRACE_ALL")) else None),
                     _inject=(_inject if it == 1 else None),
                     _max_pose=(_max_pose if it == 1 else None),
                     _viz=_viz, _viz_it=it)
        _scan_poses, _ni0 = subXY.shape[0], numItr
        subXY, numItr = _inner_loop(subXY, state, s, a)
        _iter_stats.append((it, _scan_poses, subXY.shape[0], numItr - _ni0))
        crackGen = state["crackGen"]; crackGGen = state["crackGGen"]
        BW3 = state["BW3"]; BW_working = state["BW_working"]
        mBW = state["mBW"]; nBW = state["nBW"]

        if _iter_trace is not None:
            _iter_trace.append(dict(
                it=it, fw_in=_fw_in,                       # (x,y) free space entering iter
                subcritP=np.atleast_2d(subcritP).copy(),   # (row,col) as used downstream
                ncells=len(splitReg_work),
                reebEdge=_it_reebEdge, path=_it_path, ind=_it_ind, ccPath=_it_ccPath,
                raw_subXY=_it_raw_subXY,                   # (x,y) Boustrophedon sweep, pre-fill
                subXY=np.atleast_2d(subXY)[:, :2].copy(),  # (x,y) post-fill
                numItr=numItr))

        # ---- subtract covered region ----
        obj = pu.polybuffer(np.column_stack([subXY[:, 1], subXY[:, 0]])[:, ::-1], s, kind="lines")
        bd = obj.boundary()
        px, py = bd[:, 0], bd[:, 1]
        if not (px[0] == px[-1] and py[0] == py[-1]):
            px = np.append(px, px[0]); py = np.append(py, py[0])
        dec = DecimatePoly(np.column_stack([px, py]), [1, 1], False)
        dec = dec[0] if isinstance(dec, tuple) else dec
        obj = pu.polyshape(np.asarray(dec)[:, 0], np.asarray(dec)[:, 1])
        # A near-degenerate concave mitre in the buffer/decimate chain can spike
        # a vertex to +/-Inf; strip it so the subtract below (and the downstream
        # _polyclean/_regJoin union) don't hit GEOS "orientationIndex NaN/Inf".
        obj = pu.sanitize(obj)

        final = pu.sanitize(pu.subtract(final, obj))
        final_work = pu.sanitize(pu.subtract(final_work, obj))
        splitReg_work = [pu.sanitize(pu.subtract(c, obj)) for c in splitReg_work]

        PathEdge = np.vstack([PathEdge, subXY[:, :2]])
        final = _polyclean(final)
        if not isinstance(final, pu.PolyShape):
            final = _regJoin(final) if isinstance(final, list) else final

        splitReg_work = _polyclean(splitReg_work)
        if len(splitReg_work) == 1:
            final_work = splitReg_work[0]
            final_work = _polyclean(final_work)
            if isinstance(final_work, list):
                final_work = _regJoin(final_work)
        else:
            if splitReg_work:
                final_work = _regCombine(splitReg_work)
        final_work = _polyclean(final_work)
        if isinstance(final_work, list):
            final_work = _regJoin(final_work)

        critP = PathEdge[-1].copy()
        loop1 = False

    pathLength = total_length(_rmmissing(PathEdge[1:]))
    oscclen = pxinMap(pathLength) / 12
    areaCover = 2 * pxinMap(s / math.sqrt(2)) / 12 * oscclen
    res = np.round([numItr, _DEN[dd - 1], 0.0, oscclen, areaCover], 3)
    if LOG.enabled:
        LOG.section("RESULT")
        LOG.kv("main-loop iterations", _iters)
        LOG.kv("crack-detect poses", int(numItr))
        LOG.kv("map density", f"{int(_DEN[dd - 1])}%")
        LOG.kv("path length", f"{oscclen:.3f} ft")
        LOG.kv("area covered", f"{areaCover:.3f} sq ft")
        LOG.kv("elapsed", f"{_time.time() - _t_run:.1f} s")
        if LOG.path:
            LOG.kv("log saved", LOG.path)
        if _iter_stats:                               # per-iteration table (horizontal)
            LOG.info("")
            lab_w, cell_w = 13, 6
            iters = [s[0] for s in _iter_stats]
            totals = [s[2] for s in _iter_stats]      # total (path) poses per iteration
            bar = "    +" + "-" * lab_w + "+" + "+".join("-" * cell_w for _ in iters) + "+"

            def _row(label, vals):
                return ("    |" + f"{label:^{lab_w}}" + "|"
                        + "|".join(f"{v:^{cell_w}}" for v in vals) + "|")
            LOG.info(bar)
            LOG.info(_row("main iter", iters))
            LOG.info(bar)
            LOG.info(_row("total poses", totals))
            LOG.info(bar)
        LOG.info("")
    if _viz is not None and hasattr(_viz, "finalize"):
        _viz.finalize()                              # VizOptions: render/display per mode
    if _out is not None:                             # non-invasive metric capture (sweep scoring)
        _out["WSarea"] = WSarea
        _out["coverPercent"] = (areaCover / WSarea) if WSarea else float("nan")
    return PathEdge, res


def _rmmissing(a):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    return a[~np.isnan(a).any(axis=1)]


def _prune_stray(subcritP, splitReg_work, reebEdge, s):
    """Prune stray (rMiss: not in any edge) and degree-2 (rCont) critical points by
    merging their incident cells."""
    N = subcritP.shape[0]
    nodes_in_edges = np.unique(reebEdge.ravel()) if reebEdge.size else np.array([], int)
    rMiss = np.array([i for i in range(N) if i not in set(nodes_in_edges.tolist())], dtype=int)

    adj = np.zeros((N, N))
    for u, v in reebEdge:
        adj[u, v] += 1
        adj[v, u] += 1
    deg = adj.sum(axis=0)
    rCont = np.array([i for i in range(N) if deg[i] == 2], dtype=int)
    rCont = np.array([i for i in rCont if i not in set(rMiss.tolist())], dtype=int)

    if rMiss.size == 0 and rCont.size == 0:
        return subcritP, splitReg_work, reebEdge

    r_rmin = []
    reebEdge = reebEdge.copy()
    splitReg_work = list(splitReg_work)

    for i in range(rMiss.size):
        rdis = spdist(subcritP[rMiss[i]], subcritP)
        rdis[rdis == 0] = np.inf
        rmin_i = int(np.argmin(rdis))
        if np.any(rdis < s):
            # edges incident to rmin_i
            inc_mask = (reebEdge == rmin_i).any(axis=1)
            inc_edges = reebEdge[inc_mask]
            rAreaN = np.flatnonzero(_ismember_rows(reebEdge, inc_edges))
            if rAreaN.size < 3 and rAreaN.size:
                rNodeN = reebEdge[rAreaN]
                if rNodeN.shape[0] > 1:
                    keep = ~np.isin(np.arange(reebEdge.shape[0]), rAreaN)
                    merged_cells = [splitReg_work[k] for k in rAreaN]
                    reebEdge = reebEdge[keep]
                    r_rmin.append(rmin_i)
                    rArea = pu.union(*merged_cells)
                    splitReg_work = [c for k, c in enumerate(splitReg_work) if k not in set(rAreaN.tolist())]
                    splitReg_work.append(pu.polybuffer(pu.polybuffer(rArea, 1), -1))
                    other = rNodeN[rNodeN != rmin_i]
                    reebEdge = np.vstack([reebEdge, other.reshape(1, -1)]) if other.size == reebEdge.shape[1] else reebEdge

    for i in range(rCont.size):
        inc_mask = (reebEdge == rCont[i]).any(axis=1)
        inc_edges = reebEdge[inc_mask]
        rAreaN = np.flatnonzero(_ismember_rows(reebEdge, inc_edges))
        if rAreaN.size:
            rNodeN = reebEdge[rAreaN]
            if rNodeN.shape[0] > 1:
                keep = ~np.isin(np.arange(reebEdge.shape[0]), rAreaN)
                merged_cells = [splitReg_work[k] for k in rAreaN]
                reebEdge = reebEdge[keep]
                r_rmin.append(rCont[i])
                rArea = pu.union(*merged_cells)
                splitReg_work = [c for k, c in enumerate(splitReg_work) if k not in set(rAreaN.tolist())]
                splitReg_work.append(pu.polybuffer(pu.polybuffer(rArea, 1), -1))
                other = rNodeN[rNodeN != rCont[i]]
                reebEdge = np.vstack([reebEdge, other.reshape(1, -1)]) if other.size == reebEdge.shape[1] else reebEdge

    drop = np.unique(np.concatenate([rMiss, rCont, np.array(r_rmin, dtype=int)])) if (rMiss.size or rCont.size or r_rmin) else np.array([], int)
    # NOTE: deleting subcritP rows renumbers nodes; reebEdge indices must be remapped.
    keep_nodes = np.array([i for i in range(N) if i not in set(drop.tolist())], dtype=int)
    remap = -np.ones(N, dtype=int)
    remap[keep_nodes] = np.arange(keep_nodes.size)
    subcritP = subcritP[keep_nodes]
    if reebEdge.size:
        reebEdge = remap[reebEdge]
    return subcritP, splitReg_work, reebEdge


def _start_node(reebEdge, reebCell, splitReg_work, subcritP, PathEdge):
    """Pick the starting node ind nearest the last point in PathEdge."""
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([tuple(e) for e in reebEdge])
    ncomp = nx.number_connected_components(G) if G.number_of_nodes() else 1
    last = PathEdge[-1]
    # nearest cell to the last PathEdge point, then its reeb nodes, pick by x-distance
    mind = [np.min(spdist(last, c.Vertices[~np.isnan(c.Vertices).any(axis=1)])) for c in splitReg_work]
    cell_ind = int(np.argmin(mind))
    rc = reebEdge[reebCell == cell_ind].ravel()
    if rc.size == 0:
        return 0
    k = int(np.argmin(np.abs(last[0] - subcritP[rc, 1])))
    return int(rc[k])


def _start_candidates(reebEdge, reebCell, splitReg_work, subcritP, PathEdge, eps=10.0):
    """Return candidate start nodes for global-shortest-path start selection.

    When the robot sits at the centre of its just-covered disk the uncovered cells
    are all roughly equal distance away, producing a near-tie in the nearest-cell
    selection. Returns every cell within ``eps`` of the nearest, and the caller
    keeps the one that minimises the global Boustrophedon_CellCon path length.
    A clear winner returns a single candidate, identical to ``_start_node``."""
    last = np.asarray(PathEdge)[-1]
    reebCell = np.asarray(reebCell)
    mind = np.array([np.min(spdist(last, c.Vertices[~np.isnan(c.Vertices).any(axis=1)]))
                     for c in splitReg_work], dtype=float)
    if mind.size == 0:
        return [0]
    out = []
    for cell_ind in np.flatnonzero(mind <= mind.min() + eps):
        rc = np.atleast_2d(reebEdge)[reebCell == int(cell_ind)].ravel()
        if rc.size == 0:
            continue
        k = int(np.argmin(np.abs(last[0] - subcritP[rc, 1])))
        node = int(rc[k])
        if node not in out:
            out.append(node)
    return out or [_start_node(reebEdge, reebCell, splitReg_work, subcritP, PathEdge)]


def _inner_loop(subXY, st, s, a):
    """The per-pose sensing/detection/fill loop."""
    crackGen = st["crackGen"]; crackGGen = st["crackGGen"]
    BW3 = st["BW3"]; BW_working = st["BW_working"]
    mBW = st["mBW"]; nBW = st["nBW"]; PathEdge = st["PathEdge"]
    numItr = st["numItr"]
    rowBW, colBW = BW3.shape

    intPoints = np.zeros_like(BW3)
    fl = False
    fflg = False
    d = 0
    d_lock = 0
    flag = 0
    ww = np.empty((0, 2))
    preCrack = None
    have_preCrack = False
    mBWa = np.array(mBW); nBWa = np.array(nBW)

    import os as _os, sys as _sys, time as _time
    _t0 = _time.time()
    _trace = st.get("_trace")          # if provided, append per-pose state
    _max_pose = st.get("_max_pose")    # debug cap on pose count

    # The sensor disk is a fixed s-radius circle re-centred each pose. Tessellate it
    # ONCE at the origin; per pose translate V0 to the current center (adding the center
    # offset is exact because the buffer vertices are center + s*(cos,sin) at fixed angles)
    # -- avoids a full Shapely buffer call every pose.
    _disk_V0 = pu.polybuffer((0.0, 0.0), s, kind="points").Vertices

    if LOG.enabled:
        LOG.info("")
        LOG.info("  inner loop  ----  (sense within s; fill = crack within footprint a)")
    i = 0
    while i < subXY.shape[0]:
        if _max_pose is not None and i >= _max_pose:
            break
        curPt = subXY[i, :2]
        if LOG.enabled:
            # fill = the footprint (radius a) is over a crack at this pose (dispensing),
            # i.e. subXY's fill flag -- NOT the sensor-range crack detection.
            LOG.info(f"    pose {i:>4}   (x={curPt[0]:>5.0f}, y={curPt[1]:>5.0f})   "
                     f"numItr={numItr:>4}   fill={str(bool(subXY[i, 2])):>5}   "
                     f"t={_time.time() - _t0:>6.1f}s")
        if subXY[i, 2]:
            rows, cols = _sMaskid(BW_working, a, curPt)
            # bww tracking omitted (only affects the dead remove-crack 3-arg branch)

        # sense within s. Prefilter the skeleton points to the disk's bbox before
        # the (costly) inpolygon -- points outside the bbox are outside the disk,
        # so the result is identical.
        center = curPt[::-1]                       # curPt is (x,y); disk center needs (y,x)
        V = _disk_V0 + center                       # precomputed unit disk translated to current pose
        ylo, yhi = V[:, 0].min(), V[:, 0].max()
        xlo, xhi = V[:, 1].min(), V[:, 1].max()
        cand = np.flatnonzero((mBWa >= ylo) & (mBWa <= yhi) & (nBWa >= xlo) & (nBWa <= xhi))
        sel = np.empty(0, dtype=int)
        if cand.size:
            IN, _ = pu.inpolygon(mBWa[cand], nBWa[cand], V[:, 0], V[:, 1])
            sel = cand[np.flatnonzero(IN)]
        for bwh in sel:
            crackGen[mBWa[bwh], nBWa[bwh]] = 1
            crackGGen[mBWa[bwh], nBWa[bwh]] = 1
        m = mBWa[sel]; n = nBWa[sel]

        _viz = st.get("_viz")
        if _viz is not None:
            _viz_emit(_viz, dict(kind="pose", it=st.get("_viz_it"), i=i,
                                 curPt=np.asarray(curPt, float).copy(),    # (x,y)
                                 sensed=np.column_stack([m, n]).astype(int) if len(m) else np.empty((0, 2), int),
                                 subXY=np.asarray(subXY[:, :2], float).copy(),   # evolving subXY (x,y) -> #3
                                 fill=bool(subXY[i, 2])))
            if hasattr(_viz, "on_pose"):
                _viz.on_pose(st.get("_viz_it"), i, curPt)

        # Skeleton clean loop: iteratively remove branch points then isolated single
        # pixels, accumulating branch points in intPoints (persistent across poses)
        # and mirroring removals into BW3, until a pass changes nothing.
        # Operates on a sparse foreground list `fg` instead of a full-image scan;
        # intPoints removal is a full-image mask op (must clear historical branch
        # points from each freshly sensed patch, not just the current pose).
        I_te = crackGen.copy()
        fg = argwhere2d(I_te)                      # ONE argwhere/pose: foreground (fast)
        while True:
            if fg.size:
                bpm = _neighbor_counts_at(I_te, fg) > 2  # branch points (pass-start state)
                if bpm.any():
                    bp = fg[bpm]
                    intPoints[bp[:, 0], bp[:, 1]] = 1    # accumulate (only >0 used)
            imask = intPoints > 0                  # all accumulated intersection pts
            I_te[imask] = 0
            BW3[imask] = 0
            if fg.size:
                keepm = intPoints[fg[:, 0], fg[:, 1]] == 0    # survives intPoints removal
                fg = fg[keepm]                     # foreground after intPoints removal
                removed_int = not keepm.all()
            else:
                removed_int = False
            n_px = 0
            if fg.size:
                pxm = _neighbor_counts_at(I_te, fg) == 0      # single pixels of I_te
                n_px = int(pxm.sum())
                if n_px:
                    px = fg[pxm]
                    I_te[px[:, 0], px[:, 1]] = 0
                    BW3[px[:, 0], px[:, 1]] = 0
                    fg = fg[~pxm]
            if not (removed_int or n_px):          # nothing removed this pass -> converged
                break
        crackGen = I_te

        # `fg` is exactly crackGen's foreground (maintained through the clean loop)
        # -> hand it to endP_ident to skip its full-image bool-conv + argwhere on
        # crackGen. crackGen is not modified between here and the line-776 call, so
        # `fg` stays valid for both (NOT the line-831 call, after reskeletonize).
        eP, rP, _ = endP_ident(crackGen, BW3, bw3_fg=fg)
        endlogi = np.zeros((0, 2))
        line = np.empty((0, 4))
        if np.atleast_2d(rP).shape[0] > 1:
            _, line, _, _ = compCrack(crackGen, eP, _DIR_MAP, None, fg=fg)   # fg = crackGen foreground
            line = np.atleast_2d(line)
            starts = line[:, [0, 1]]; ends = line[:, [2, 3]]
            allp = np.vstack([starts, ends])
            K = line.shape[0]
            flagv = np.array([np.any(spdist(p, np.atleast_2d(rP)) < 5) for p in allp])
            endlogi = np.column_stack([flagv[:K], flagv[K:]])

        # The sensed crack pixels must span BOTH multiple rows AND multiple columns
        # (genuine 2D extent). A 1-D line of pixels does not constitute a fillable crack.
        cond_a = (m.size and (not np.all(m == m[0])) and (not np.all(n == n[0])))
        cond_b = (endlogi.size and endlogi.sum(axis=1).max() == 2
                  and line.size and np.any(spdist2(line[:, :2], line[:, 2:4]) > a))
        _inj = st.get("_inject")
        _inj_i = _inj[i] if (_inj is not None and i < len(_inj)) else None
        _replay = _inj_i is not None                 # inject present (WP override)
        _force = _replay and len(_inj_i) > 2         # 4-tuple => full replay (force detection)
        do_fill = bool(_inj_i[2]) if _force else (cond_a or cond_b)
        if do_fill:
            eP, rP, cP = endP_ident(crackGen, BW3, bw3_fg=fg)
            rPa = np.atleast_2d(rP)
            # ---- remove already-filled cracks ----
            if (not _force) and rPa.shape[0] > 1 and ww.shape[0]:
                crackRaw, _, _, _ = compCrack(crackGen, eP, _DIR_MAP, None, fg=fg)
                ww1 = np.flatnonzero(_ismember_rows(ww, subXY[i, :2][None, :])).size > 0
                if ww1:
                    intc = argwhere2d(intPoints > 0)         # (row,col)
                    pp = pu.polybuffer(subXY[:i + 1, :2], a + 25, kind="lines")
                    for c in range(len(crackRaw)):
                        cr = np.atleast_2d(np.asarray(crackRaw[c], float))
                        ends = cr[[0, -1]]
                        cond1 = _ismember_rows(ends, rPa).sum() == 2
                        cond2 = np.flatnonzero((spdist(cr[0], rPa) < a) | (spdist(cr[-1], rPa) < a)).size == 2
                        if not (cond1 or cond2):
                            continue
                        inn = pu.isinterior(pp, ends[:, ::-1])
                        cq = pu.isinterior(pp, cr[:, ::-1])
                        c_chq = cq.sum() / len(cq) * 100
                        if _os.environ.get("OSCC_CLEANUP_DBG"):
                            # diagnose dense-map non-termination: which crack is
                            # re-detected, and at what coverage it plateaus below the gate.
                            print("  [cleanup] i=%d c=%d len=%d inn=%d c_chq=%.1f%% ends=%s fired=%s"
                                  % (i, c, cr.shape[0], int(inn.sum()), c_chq,
                                     np.round(ends.ravel(), 0).tolist(),
                                     bool(inn.sum() == 2 and c_chq > 90)), flush=True)
                        if inn.sum() == 2 and c_chq > 90:
                            post = []
                            for ee in range(2):
                                r0, c0 = int(ends[ee, 0]), int(ends[ee, 1])
                                post.append(int(crackGen[r0-1, c0-1] + crackGen[r0-1, c0] + crackGen[r0-1, c0+1]
                                                + crackGen[r0, c0-1] + crackGen[r0, c0+1]
                                                + crackGen[r0+1, c0-1] + crackGen[r0+1, c0] + crackGen[r0+1, c0+1]))
                            cri = cr.astype(int)
                            crackGen[cri[:, 0], cri[:, 1]] = 0
                            BW_working[cri[:, 0], cri[:, 1]] = 0
                            BW3[cri[:, 0], cri[:, 1]] = 0
                            keepb = ~_ismember_rows(np.column_stack([mBWa, nBWa]), cr)
                            mBWa = mBWa[keepb]; nBWa = nBWa[keepb]
                            fl = True
                            if intc.size:
                                inint = pu.isinterior(pp, intc[:, ::-1])
                                for (r0, c0) in intc[inint]:
                                    BW3[r0, c0] = 0
                            spx, _ = neighbor_count_points(BW3, 0)          # bwmorph 'clean'
                            BW3[spx > 0] = 0
                            for ee in range(2):
                                if post[ee] > 1:
                                    r0, c0 = int(ends[ee, 0]), int(ends[ee, 1])
                                    crackGen[r0, c0] = 1; BW_working[r0, c0] = 1; BW3[r0, c0] = 1
                            stt = curPt[::-1]
                            from skimage.morphology import skeletonize as _skel
                            crackGen = _skel(crackGen > 0).astype(int)  # re-thin to a 1-px skeleton
                            fflg = True
                            d_lock = None
                            eP, rP, cP = endP_ident(crackGen, BW3)
            # ----
            if not subXY[i, 2]:
                stt = curPt[::-1]
            else:
                # stt is not updated on a fill waypoint (subXY[i,2]==1) -- it stays
                # at the fill-start (last scanning pose) so image_planning routes
                # from the correct origin.
                d_lock = (d_lock + 1) if d_lock is not None else None

            if _force:
                # Replay mode: use the injected WP/flag directly, skipping image_planning.
                WP = np.atleast_2d(_inj_i[0].copy()); flag = int(_inj_i[1])
                numItr += 1
            else:
                ppath = subXY[:i + 1, :2]
                pre_in = preCrack if have_preCrack else None
                WP, flag, preCrack, _ = ImagePlanning_oSCC(
                    crackGen, a, s, stt, curPt[::-1], eP, rP, cP, ppath, pre_in)
                have_preCrack = True
                numItr += 1
                WP = np.atleast_2d(WP)
                if _replay:                          # replay injection: override WP only
                    WP = np.atleast_2d(_inj_i[0].copy()); flag = int(_inj_i[1])

            if flag:
                if _force and _inj_i[3] is not None:
                    d = int(_inj_i[3])       # replay: use injected resume index
                elif i == 0:
                    d = 1
                else:
                    # Restart the fill sequence (d=1) whenever the previous waypoint
                    # was a scanning pose (subXY[i-1,2]==0) or after a crack-cleanup; otherwise resume.
                    if (not subXY[i - 1, 2]) or fl:
                        d = 1; fl = False
                    else:
                        d = d + 1
                subXY = subXY[:i + 1]
                WPflip = np.column_stack([WP[:, 1], WP[:, 0]])    # WP is (row,col); convert to (x,y)
                _last = subXY[-1, :2]
                _br = ""
                # Exact (not tolerant) row membership: a curPt that is merely close
                # to a WP vertex must not match and fall through to the projection branch.
                _exact = np.all(WPflip == _last, axis=1)
                if d == 1:
                    add = np.column_stack([WPflip[d - 1:], WP[d - 1:, 2]])
                    _br = "d1"
                elif _exact.any():
                    iw = int(np.flatnonzero(_exact)[0]) + 1
                    add = np.column_stack([WPflip[iw:], WP[iw:, 2]])
                    _br = f"mem iw={iw}"
                else:
                    # Forward-project the robot onto the WP polyline: find the nearest
                    # segment and resume from its forward end vertex; if the robot
                    # projects onto the segment's first half (t<=0.5) resume from the
                    # start vertex instead, so a robot sitting behind WP[0] keeps WP[0].
                    if WPflip.shape[0] < 2:
                        mx = 0
                        _br = "proj single"
                    else:
                        A = WPflip[:-1]; B = WPflip[1:]
                        AB = B - A
                        denom = (AB * AB).sum(1)
                        denom[denom == 0] = 1.0            # zero-length (duplicate) seg
                        tt = ((_last - A) * AB).sum(1) / denom
                        ttc = np.clip(tt, 0.0, 1.0)
                        proj = A + ttc[:, None] * AB
                        dseg = ((proj - _last) ** 2).sum(1)
                        k = int(np.argmin(dseg))
                        mx = k + 1 if tt[k] > 0.5 else k
                        _br = f"proj k={k} t={tt[k]:.2f} mx={mx}"
                    add = np.column_stack([WPflip[mx:], WP[mx:, 2]])
                if _os.environ.get("OSCC_SPLICE_DBG"):
                    print(f"  [splice] i={i} d={d} br={_br} |WP|={WPflip.shape[0]} |add|={add.shape[0]} "
                          f"last={np.round(_last,1)}", flush=True)
                subXY = np.vstack([subXY, add])
                ww = WPflip
            else:
                if d_lock == 1:
                    d = 1
                else:
                    d = d + 1
        else:
            # ---- ELSE-branch: remove already-filled cracks ----
            # When the fill condition is false but the robot is re-passing a
            # previously-filled crack, strip it from crackGen/BW_working and the
            # master skeleton list mBW/nBW so it is not re-sensed on later poses.
            # Lighter than the fill-branch cleanup: only checks that both ends are
            # real endpoints (cond1) and that both lie inside the path buffer (no c_chq);
            # does not touch BW3/intPoints.
            if (not _force) and np.atleast_2d(rP).shape[0] > 1 and ww.shape[0]:
                rPa = np.atleast_2d(rP)
                crackRaw, _, _, _ = compCrack(crackGen, eP, _DIR_MAP, None, fg=fg)
                ww1 = np.flatnonzero(_ismember_rows(ww, subXY[i, :2][None, :])).size > 0
                if ww1:
                    pp = pu.polybuffer(subXY[:i + 1, :2], a + 25, kind="lines")
                    for c in range(len(crackRaw)):
                        cr = np.atleast_2d(np.asarray(crackRaw[c], float))
                        ends = cr[[0, -1]]
                        if _ismember_rows(ends, rPa).sum() != 2:        # cond1 only
                            continue
                        inn = pu.isinterior(pp, ends[:, ::-1])
                        if inn.sum() == 2:
                            post = []
                            for ee in range(2):
                                r0, c0 = int(ends[ee, 0]), int(ends[ee, 1])
                                post.append(int(crackGen[r0-1, c0-1] + crackGen[r0-1, c0] + crackGen[r0-1, c0+1]
                                                + crackGen[r0, c0-1] + crackGen[r0, c0+1]
                                                + crackGen[r0+1, c0-1] + crackGen[r0+1, c0] + crackGen[r0+1, c0+1]))
                            cri = cr.astype(int)
                            crackGen[cri[:, 0], cri[:, 1]] = 0
                            BW_working[cri[:, 0], cri[:, 1]] = 0    # NB: BW3 is not updated in this lighter branch
                            keepb = ~_ismember_rows(np.column_stack([mBWa, nBWa]), cr)
                            mBWa = mBWa[keepb]; nBWa = nBWa[keepb]
                            fl = True
                            for ee in range(2):
                                if post[ee] > 1:
                                    r0, c0 = int(ends[ee, 0]), int(ends[ee, 1])
                                    crackGen[r0, c0] = 1; BW_working[r0, c0] = 1
                            stt = curPt[::-1]
                            from skimage.morphology import skeletonize as _skel
                            crackGen = _skel(crackGen > 0).astype(int)  # re-thin to a 1-px skeleton
                            fflg = True

        if _trace is not None:
            _allm = bool(m.size and np.all(m == m[0]))
            _alln = bool(m.size and np.all(n == n[0]))
            _trace.append(dict(it=st.get("_viz_it"), i=i, n=subXY.shape[0], d=d, flag=int(flag),
                               numItr=numItr, fflg=bool(fflg),
                               dlock=(-1 if d_lock is None else d_lock),
                               cg=int((crackGen > 0).sum()),
                               cg_rc=np.argwhere(crackGen > 0),
                               wp=np.atleast_2d(ww).copy(),
                               subxy=subXY[:, :2].copy(),
                               last=subXY[-1, :3].copy(),
                               curPt=np.asarray(curPt, float).copy(),
                               n_m=int(m.size), all_m_same=int(_allm), all_n_same=int(_alln),
                               sizerP=int(np.atleast_2d(rP).shape[0]) if np.size(rP) else 0,
                               condA=int(bool(cond_a)), condB=int(bool(cond_b)),
                               do_fill=int(bool(do_fill))))
        i += 1
        # When reskeletonize (fflg) occurs on or after the last scan pose, append
        # a duplicate of the final waypoint so the inner loop processes it. The
        # guard `i >= shape[0]-1` fires before processing the last row (0-based),
        # which is the correct point to append the duplicate.
        if i >= subXY.shape[0] - 1 and fflg:
            subXY = np.vstack([subXY, subXY[-1]])
            fflg = False

    st.update(crackGen=crackGen, crackGGen=crackGGen, BW3=BW3, BW_working=BW_working,
              mBW=list(mBWa), nBW=list(nBWa), PathEdge=PathEdge)
    return subXY, numItr


if __name__ == "__main__":
    import argparse

    class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass                                              # show defaults + keep epilog newlines

    p = argparse.ArgumentParser(
        prog="OnlineSCC.py",
        formatter_class=_Fmt,
        description="OnlineSCC: complete-coverage scan + autonomous crack-fill planner for a "
                    "crack map. Runs the planner and (optionally) renders the per-step animation.",
        epilog="examples:\n"
               "  python OnlineSCC.py                         # headless: logs progress + res\n"
               "  python OnlineSCC.py -q                      # silence the progress logging\n"
               "  python OnlineSCC.py myCrack8_100_1 --viz save --style color --smooth 0.07\n"
               "  python OnlineSCC.py --viz live --style color\n\n"
               "Robot dimensions (base / footprint / sensor diameters) are configured in\n"
               "robot_config.json -- edit that file to change the robot.",
    )
    p.add_argument("map", nargs="?", default=None,
                   help="explicit crack-map name (e.g. myCrack8_100_1). Overrides "
                        "--den/--sig/--map-num; omit it to build the map from those flags.")
    p.add_argument("--den", type=int, default=100, choices=_DEN, metavar="PCT",
                   help="crack density %% (one of 35/45/50/65/80/90/95/100; default 100)")
    p.add_argument("--sig", type=int, default=None, choices=[5, 10, 20], metavar="S",
                   help="Gaussian sigma (5/10/20). If given -> a GAUSSIAN map; omitted -> UNIFORM.")
    p.add_argument("--map-num", dest="mapnum", type=int, default=None, metavar="N",
                   help="map number: Uniform -> variant 1-5 (random if omitted); "
                        "Gaussian -> set folder 1-6 (random if omitted).")
    p.add_argument("--max-iter", type=int, default=None, help="cap outer iterations (debug)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="silence the CONSOLE log (the logs/<algo>_<map>.log file is still "
                        "written). Sectioned progress logging is ON by default -- far cheaper "
                        "than --viz (no per-pose capture/render) and doesn't perturb timing")

    g = p.add_argument_group("visualization", "omit --viz to run headless (no rendering)")
    g.add_argument("--viz", choices=["show", "save", "show+save", "live"],
                   help="output mode: live=plot ON SCREEN as it runs (needs a GUI backend); "
                        "show=open viewer at end; save=write GIF+PNGs; show+save=both. "
                        "The single final-path PNG is ALWAYS written to Results/OnlineSCC/ (every "
                        "mode); GIFs go to Results/GIF/. OMIT --viz: just the final-path PNG. "
                        "(For live text progress use the default logging, not --viz.)")
    g.add_argument("--style", choices=["plain", "publish", "color"], default="plain",
                   help="render style: plain / publish (detailed: Reeb graph, labels, callouts) / "
                        "color (detailed + colour cells). Saving publish/color -> hi-res. "
                        "e.g. --viz save --style color, or --viz live --style color")
    g.add_argument("--step", type=int, default=1, help="poses per coverage frame (fill poses always shown)")
    g.add_argument("--fps", type=int, default=14, help="GIF frames per second")
    g.add_argument("--smooth", type=float, default=0.0, metavar="M",
                   help="smooth robot motion: tween step in metres (e.g. 0.07); 0 = off")
    g.add_argument("--out", default=None, metavar="GIF", help="output GIF path "
                   "(default: Results/GIF/<map>_unKnown.gif)")
    g.add_argument("--frames", default=None, metavar="DIR",
                   help="also write full-res per-step PNGs to this dir")
    g.add_argument("--gif-max-px", type=int, default=1100, help="cap the GIF's larger side (px)")
    args = p.parse_args()

    from private.mapselect import resolve_map
    img_n, dd, desc = resolve_map(args.map, den=args.den, sig=args.sig, mapnum=args.mapnum)
    _tag = img_n.replace("/", "_")

    # logging ON by default: console (unless -q) + a rotating file logs/OnlineSCC_<map>.log
    _logpath = rotate_path("logs", f"OnlineSCC_{_tag}")
    LOG.enable(console=not args.quiet, logfile=_logpath)
    print(f"map: {desc}")

    from private.visualize import VizOptions
    # the single final-path PNG (written in EVERY mode) -> Results/OnlineSCC/; GIFs -> Results/GIF/.
    # --out overrides the matching output by extension (.png -> the PNG, .gif -> the GIF).
    _final_png = (args.out if (args.out and args.out.endswith(".png"))
                  else f"Results/OnlineSCC/{_tag}_unKnown.png")
    _gif = (args.out if (args.out and args.out.endswith(".gif"))
            else f"Results/GIF/{_tag}_unKnown.gif")
    if args.viz:
        viz = VizOptions(
            mode=args.viz, style=args.style, step=args.step, fps=args.fps,
            smooth_step_m=args.smooth, gif_max_px=args.gif_max_px,
            out_gif=_gif, final_png=_final_png,
            frame_dir=args.frames, title=f"OnlineSCC: {img_n}")
    else:
        # default: auto-render just the Final-Path frame to Results/OnlineSCC (no --viz needed)
        viz = VizOptions(mode="final", style="color",
                         out_gif=_gif, final_png=_final_png,
                         title=f"OnlineSCC: {img_n}")

    try:
        PE, res = run_online_scc(img_n=img_n, dd=dd, max_iter=args.max_iter, _viz=viz)
    except KeyboardInterrupt:
        # record the interruption + traceback in the log so a stopped run is visible
        LOG.exception("KeyboardInterrupt (Ctrl-C) -- run stopped by user")
        raise
    except Exception as e:
        # any crash: record the full traceback in the log before it propagates
        LOG.exception(f"{type(e).__name__} -- run crashed")
        raise
    finally:
        LOG.disable()
    print("res =", res)
    print("PathEdge rows:", PE.shape[0])

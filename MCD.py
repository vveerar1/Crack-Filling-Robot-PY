"""Morse Cell Decomposition (MCD) of a planar free-space region.

MCD splits the robot's free-space polygon into a set of *monotone coverage
cells* — vertical strips whose boundaries are monotone in x — so that each cell
can be swept by a simple boustrophedon (zig-zag) pass.  The decomposition is
driven by the *critical points* of the boundary: locations where the boundary
has a local x-extremum (leftmost / rightmost point on the outer contour or
around each obstacle hole).  At every concave critical point, a thin vertical
*split edge* is inserted to cut the free space into two separate cells.

Algorithm outline
  1. Refine the outer boundary to a dense, evenly-spaced vertex representation
     so that flat edges have a vertex at their geometric centre.
  2. Scan the boundary cyclically for x-local-minima (forward pass) and
     x-local-maxima (backward pass); do the same for each obstacle hole.
  3. At each critical point test whether it is convex or concave relative to
     the interior.  Concave points require a vertical cut (split edge).
  4. Insert thin split-edge bars into the working polygon, then sort the
     resulting cells by centroid distance from the origin.
  5. Merge critical points that are closer than 25 px (drop the concave
     duplicate to avoid redundant cuts).

Usage
-----
Called once per outer planning iteration from the SCC and OnlineSCC drivers,
after the free-space polygon has been buffered inward by the robot footprint.
The returned cells and critical points are passed directly to :func:`Reeb`.
"""

import os as _os

import numpy as np

from shapely.geometry import LineString, Point

from private.utils import islocalmin, islocalmax, spdist
from private import poly_utils as pu
from private.poly_utils import PolyShape
from private.refinePoly import refinePoly, _insert_midpoints, _drop_closing


def _refined_boundary(boundary0, times):
    """Produce the refined boundary vertex sequence for the SCC critical-point scan.

    Each refinement pass inserts midpoints, prepends the last vertex, then rotates
    the array by ``N // 10`` positions.  The starting position after rotation determines
    the order in which x-extrema are encountered during the forward scan, which
    controls the order in which critical points are emitted for the SCC planner."""
    B = _drop_closing(np.asarray(boundary0.boundary(), float))
    for _ in range(int(times)):
        B = _insert_midpoints(B)
        B = np.vstack([B[-1:], B])
        B = np.roll(B, int(B.shape[0] // 10), axis=0)
    return B


# Forward (x-min) scan uses a resampled uniform boundary to robustly detect flat-edge
# centers regardless of vertex distribution.  Backward (x-max) scan uses the raw
# refined boundary so that smoothing-arc apexes survive as strict maxima; in legacy
# mode (OSCC_MCD_LEGACY_SCAN=1) both scans use the resampled boundary.
_EMIT_LOG = []   # per-region critical-emission log (gated by OSCC_MCD_EMIT env var)
_MATLAB_START = bool(_os.environ.get("OSCC_MCD_MATLAB_START"))
_LEGACY_SCAN = bool(_os.environ.get("OSCC_MCD_LEGACY_SCAN"))
_FWD_RND = 0
_BWD_RND = 0 if _LEGACY_SCAN else int(_os.environ.get("OSCC_SCAN_RND", "6"))
# Orphan backward-max recovery: a fine-scan x-max whose nearest coarse-scan anchor
# `leg` is farther than this is a REAL shallow x-max in a boundary stretch the coarse
# forward scan flattened entirely (no anchor), not a refinement of an existing leg --
# keep it, else a shallow concave max is dropped, losing a critical point and a cell.
_ORPHAN_MAX = float(_os.environ.get("OSCC_ORPHAN_MAX", "50"))


def _outer_nb_fwd(boundary):
    """Forward (x-min) scan boundary: uniformly resampled for robust flat-center detection."""
    return _resample(_ring(boundary))


def _outer_nb_bwd(boundary):
    """Backward (x-max) scan boundary: raw ring (preserves smoothing-arc apexes), or resampled in legacy mode."""
    return _resample(_ring(boundary)) if _LEGACY_SCAN else _ring(boundary)


def _backward_maxima(nb_b, nb_f):
    """Backward (x-max) critical points, hybrid flat-center / arc-apex.

    A coarse scan (round-0) on the resampled boundary identifies the correct set of
    x-max regions (one flat-center each).  For each region, if exactly one fine-scan
    (round-_BWD_RND) peak is nearest to it, that peak is promoted as the critical point
    (resolves a smoothing-arc apex at the correct y).  If multiple fine peaks map to
    one coarse region, the flat-center is kept.  Genuine orphan fine peaks with no
    nearby coarse anchor (shallow concavities) are also preserved."""
    leg = nb_f[_cyclic_extrema_mask(np.round(nb_f[:, 0], 0), find_min=False)]
    if _LEGACY_SCAN or leg.shape[0] == 0:
        return leg
    fai = nb_b[_cyclic_extrema_mask(np.round(nb_b[:, 0], _BWD_RND), find_min=False)]
    if fai.shape[0] == 0:
        return leg
    assign = np.array([int(np.argmin(np.hypot(*(leg - fp).T))) for fp in fai])
    out = []
    for i, lp in enumerate(leg):
        group = fai[assign == i]
        out.append(group[0] if group.shape[0] == 1 else lp)
    out = np.array(out, dtype=float)
    # recover orphan fine-scan maxima: a fine-scan peak with no coarse anchor within
    # _ORPHAN_MAX px is a genuine shallow maximum in a stretch the coarse scan
    # flattened; keep it.
    if not _LEGACY_SCAN:
        for fp in fai:
            d_leg = np.hypot(*(leg - fp).T).min()
            d_out = np.hypot(*(out - fp).T).min() if out.size else np.inf
            if d_leg > _ORPHAN_MAX and d_out > _ORPHAN_MAX:
                out = np.vstack([out, fp])
    return out


def _collect_ys(geom):
    """All y-coordinates of a Shapely intersection result (any type)."""
    ys = []
    gt = geom.geom_type
    if geom.is_empty:
        return ys
    if gt == "Point":
        ys.append(geom.y)
    elif gt in ("MultiPoint", "GeometryCollection", "MultiLineString"):
        for g in geom.geoms:
            ys.extend(_collect_ys(g))
    elif gt == "LineString":
        ys.extend(np.asarray(geom.coords)[:, 1].tolist())
    return ys


def _scan_inside(ps, x, ymin, ymax):
    """Return the solid segments of a vertical line x=[ymin,ymax] inside polygon ps.

    Robust to lines collinear with hole edges: collects all boundary crossings,
    then classifies each y-interval via a boundary-exclusive ``contains`` test on
    its midpoint.  Returns entry/exit pairs as an (N, 2) array, NaN-row-separated
    for disconnected segments.
    """
    g = ps.geom
    line = LineString([(x, ymin), (x, ymax)])
    ys = set([float(ymin), float(ymax)])
    for y in _collect_ys(line.intersection(g.boundary)):
        ys.add(float(y))
    ys = sorted(ys)
    segs = []
    for i in range(len(ys) - 1):
        ymid = 0.5 * (ys[i] + ys[i + 1])
        if g.contains(Point(x, ymid)):
            if segs and abs(segs[-1][1] - ys[i]) < 1e-9:
                segs[-1][1] = ys[i + 1]
            else:
                segs.append([ys[i], ys[i + 1]])
    out = []
    for i, (y0, y1) in enumerate(segs):
        if i > 0:
            out.append([np.nan, np.nan])
        out.append([x, y0])
        out.append([x, y1])
    return np.array(out, dtype=float) if out else np.empty((0, 2))

_T = 0.5


def _as_list(x):
    if isinstance(x, PolyShape):
        return [x]
    return list(x)


def _ring(ps):
    """Open boundary ring (closing duplicate dropped) for cyclic scanning."""
    nb = ps.boundary()
    if nb.shape[0] >= 2 and np.allclose(nb[0], nb[-1], equal_nan=True):
        nb = nb[:-1]
    return nb


def _resample(ring, ds=1.0):
    """Resample a closed boundary ring at approximately ``ds`` arc-length spacing.

    Ensures an even vertex distribution so that flat-edge centers have a vertex
    at the true geometric midpoint, which is required for correct flat-center
    detection by the cyclic extrema scan.
    """
    ring = np.asarray(ring, dtype=float)
    if ring.shape[0] < 3 or np.isnan(ring).any():
        return ring
    pts = np.vstack([ring, ring[:1]])
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return ring
    n = max(int(np.ceil(total / ds)), ring.shape[0])
    s = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return np.column_stack([x, y])


_HOLE_REF = {}   # optional override: (kind, hi) -> (N,2) target for diagnostic forcing


def _hole_perm(idx, hv, hi, kind, matlab_start):
    """Return the permutation that places hole x-extrema in SCC emission order.

    The SCC planner traverses the buffered-hole boundary clockwise from the
    minimum-y vertex; x-extrema are emitted in that traversal order.  This
    function computes the clockwise arc position of each extremum from the min-y
    vertex and returns the sort permutation accordingly.

    Parameters
    ----------
    idx : array-like of int
        Indices into *hv* of the x-extrema found by the cyclic scan.
    hv : ndarray, shape (N, 2)
        Resampled hole boundary ring (open, no closing duplicate).
    hi : int
        Hole index (used only when a ``_HOLE_REF`` override is active).
    kind : {"min", "max"}
        Whether *idx* refers to x-minima or x-maxima of the hole.
    matlab_start : bool
        When ``True`` apply the SCC emission ordering; when ``False`` return
        ascending index order (native cyclic scan order).

    Returns
    -------
    perm : ndarray of int
        Permutation indices such that ``idx[perm]`` gives the desired order.
    """
    idx = np.asarray(idx, int)
    if idx.size == 0:
        return idx[:0]
    ref = _HOLE_REF.get((kind, hi))
    if ref is not None:
        pts = np.atleast_2d(hv[idx])
        ref = np.atleast_2d(np.asarray(ref, float))
        if pts.shape[0] == ref.shape[0]:
            perm, used = [], set()
            for t in ref:
                d = np.hypot(pts[:, 0] - t[0], pts[:, 1] - t[1])
                for j in np.argsort(d):
                    if j not in used:
                        perm.append(int(j)); used.add(int(j)); break
            return np.array(perm, int)
    if not matlab_start:
        return np.argsort(idx, kind="stable")        # native: ascending ring index
    N = hv.shape[0]
    # min-Y start vertex; ties broken by larger x
    ymin = hv[:, 1].min()
    cand = np.flatnonzero(hv[:, 1] <= ymin + 1e-9)
    miny_i = int(cand[np.argmax(hv[cand, 0])])
    key = (idx - miny_i) % N                          # CW arc position from min-Y
    return np.argsort(key, kind="stable")


def _cyclic_extrema_mask(x, find_min):
    """Cyclic x-extrema of a closed boundary, one per flat run (flat-center).

    Treats the boundary as a ring so the global extremum is never split at an
    array endpoint and each flat edge is detected exactly once.
    """
    x = np.asarray(x, dtype=float)
    M = x.size
    mask = np.zeros(M, dtype=bool)
    if M < 3:
        return mask
    changes = np.flatnonzero(x != np.roll(x, 1))
    if changes.size == 0:
        return mask
    r = int(changes[0])
    xr = np.roll(x, -r)
    runs = []
    s = 0
    for i in range(1, M):
        if xr[i] != xr[i - 1]:
            runs.append((s, i - 1, xr[s]))
            s = i
    runs.append((s, M - 1, xr[s]))
    nr = len(runs)
    for k, (rs, re, val) in enumerate(runs):
        pv = runs[(k - 1) % nr][2]
        nv = runs[(k + 1) % nr][2]
        ext = (pv > val and nv > val) if find_min else (pv < val and nv < val)
        if ext:
            center = rs + (re - rs) // 2
            mask[(center + r) % M] = True
    return mask


def _zero_nan_adjacent(TF, nb):
    """Set TF rows adjacent to NaN boundary rows to 0 (both columns)."""
    nanrows = np.flatnonzero(np.isnan(nb[:, 0]))
    if nanrows.size:
        idx = np.concatenate([nanrows + 1, nanrows - 1])
        idx = idx[(idx >= 0) & (idx < TF.shape[0])]
        TF[idx, :] = False


def _cvex_outer(inPoly, pt, find_min):
    """Convex/concave flag at boundary point via +/-5 px interior probe (outer/backward)."""
    in1 = pu.isinterior(inPoly, [pt[0] - 5, pt[1]])[0]
    in2 = pu.isinterior(inPoly, [pt[0] + 5, pt[1]])[0]
    if find_min:   # forward (minima): in1&~in2->0, ~in1&in2->1, else 1
        return 0 if (in1 and not in2) else (1 if (not in1 and in2) else 1)
    else:          # backward (maxima): in1&~in2->1, ~in1&in2->0, else 1
        return 1 if (in1 and not in2) else (0 if (not in1 and in2) else 1)


def _cvex_hole(hole, pt, is_min):
    """Convex/concave flag around a hole (note the ~isinterior)."""
    in1 = not pu.isinterior(hole, [pt[0] - 5, pt[1]])[0]
    in2 = not pu.isinterior(hole, [pt[0] + 5, pt[1]])[0]
    if is_min:     # hole minima: in1&~in2->0, ~in1&in2->1, else 1
        return 0 if (in1 and not in2) else (1 if (not in1 and in2) else 1)
    else:          # hole maxima: in1&~in2->1, ~in1&in2->0, else 1
        return 1 if (in1 and not in2) else (0 if (not in1 and in2) else 1)


def MCD(polyin_buffed, polyin_work, polyin, nodeend, flag, matlab_start=None):
    """Decompose a free-space polygon into monotone coverage cells.

    Scans the boundary of each region in *polyin_buffed* for critical points
    (x-local-extrema on the outer contour and around obstacle holes), classifies
    each as convex or concave, and inserts thin vertical split edges at every
    concave critical point to cut the polygon into monotone cells.

    Parameters
    ----------
    polyin_buffed : PolyShape or list of PolyShape
        The robot-footprint-buffered free-space polygon whose boundary is scanned
        for critical points.  May contain multiple disconnected regions.
    polyin_work : PolyShape or list of PolyShape
        The working free-space polygon into which split edges are inserted.
        Typically the same shape as *polyin_buffed* but without the buffer.
    polyin : PolyShape or list of PolyShape
        The base free-space polygon (no footprint buffer) that is also split in
        parallel.  Its cells are returned as *polyout*.
    nodeend : int or array-like
        Reserved for future use; currently unused inside the function body but
        kept in the signature for interface compatibility.
    flag : bool
        Smoothing flag passed to :func:`refinePoly`.  ``True`` applies boundary
        smoothing; ``False`` skips it.  If the smoothed boundary yields no
        critical points the function retries automatically with the flag inverted.
    matlab_start : bool or None, optional
        Selects the critical-point *emission order*.  ``True`` enables the
        boundary-scan ordering used by the offline SCC planner; ``None`` (default)
        reads the ``OSCC_MCD_MATLAB_START`` environment variable so the online
        planner's behaviour is unaffected.

    Returns
    -------
    critPT : ndarray, shape (K, 2)
        All critical points collected across every region, in (x, y) polygon
        coordinates.
    polyout_work : list of PolyShape
        Coverage cells derived from *polyin_work* after all split edges have been
        applied, sorted by centroid distance from the origin.
    polyout : list of PolyShape
        Coverage cells derived from *polyin* (the un-buffered base polygon),
        sorted in the same order as *polyout_work*.
    splitEdge : list of PolyShape
        Thin rectangular bars representing each vertical split edge inserted into
        the polygon.  Passed unchanged to :func:`Reeb` so it can expand cells
        across the cuts when searching for enclosing critical points.
    """
    # ``matlab_start=True`` selects the SCC boundary-scan emission order; ``None`` reads
    # the ``OSCC_MCD_MATLAB_START`` env var.
    ms = _MATLAB_START if matlab_start is None else bool(matlab_start)
    pb = pu.regions(polyin_buffed) if isinstance(polyin_buffed, PolyShape) else _as_list(polyin_buffed)
    if len(pb) > 1:
        pw = pu.regions(polyin_work) if isinstance(polyin_work, PolyShape) else _as_list(polyin_work)
    else:
        pw = _as_list(polyin_work)

    critPT = np.empty((0, 2))
    splitEdge = []

    for p in range(len(pb)):
        boundary = pu.rmholes(pb[p])
        critP = np.empty((0, 2))
        cvex = []

        # ---- forward boundary scan (x-minima), cyclic; retry less smoothing ----
        boundary0 = boundary                             # pre-refine copy for SCC emission order
        boundary = refinePoly(boundary, 2, flag)
        inPoly = pu.polyshape(boundary.boundary())
        nb_b = _outer_nb_bwd(boundary)
        if ms:                                           # SCC emission order: cyclic rotation to first x-min
            nb_f = _refined_boundary(boundary0, 2)
            _m0 = _cyclic_extrema_mask(np.round(nb_f[:, 0], _FWD_RND), find_min=True)
            _fi = np.flatnonzero(_m0)
            if _fi.size:
                nb_f = np.roll(nb_f, int(_fi[0] + 1), axis=0)     # rotate boundary to start at the first minimum
        else:
            nb_f = _outer_nb_fwd(boundary)
        mask_min = _cyclic_extrema_mask(np.round(nb_f[:, 0], _FWD_RND), find_min=True)
        if not mask_min.any():   # retry without smoothing
            cvex = []
            boundary = refinePoly(pu.rmholes(pb[p]), 2, not flag)
            inPoly = pu.polyshape(boundary.boundary())
            nb_f = _outer_nb_fwd(boundary)
            nb_b = _outer_nb_bwd(boundary)
            mask_min = _cyclic_extrema_mask(np.round(nb_f[:, 0], _FWD_RND), find_min=True)
        fwd_idx = np.flatnonzero(mask_min)
        subcritP = nb_f[fwd_idx]
        for j in fwd_idx:
            cvex.append(_cvex_outer(inPoly, nb_f[j], find_min=True))
        critP = np.vstack([critP, subcritP])
        if _os.environ.get("OSCC_MCD_EMIT"):
            _EMIT_LOG.append(("fwd", p, np.array(subcritP, float),
                              np.array(boundary.boundary(), float)))

        # ---- holes boundary scan ----
        hole_objs = pu.holes(pb[p])
        hole_objs = [pu.polybuffer(pu.polybuffer(h, 10), -10, joint="miter", miter_limit=4)
                     for h in hole_objs]
        _hi = 0
        for hole in hole_objs:
            for reg in pu.regions(hole):
                _hi += 1
                hv = _resample(_ring(reg))
                # x-minima (left extreme of obstacle)
                idx_min = np.flatnonzero(_cyclic_extrema_mask(np.round(hv[:, 0], 0), find_min=True))
                idx_min = idx_min[_hole_perm(idx_min, hv, _hi, "min", ms)]
                critP = np.vstack([critP, hv[idx_min]])
                for j in idx_min:
                    cvex.append(_cvex_hole(reg, hv[j], is_min=True))
                # x-maxima (right extreme of obstacle)
                idx_max = np.flatnonzero(_cyclic_extrema_mask(np.round(hv[:, 0], 0), find_min=False))
                idx_max = idx_max[_hole_perm(idx_max, hv, _hi, "max", ms)]
                critP = np.vstack([critP, hv[idx_max]])
                for j in idx_max:
                    cvex.append(_cvex_hole(reg, hv[j], is_min=False))
                if _os.environ.get("OSCC_MCD_EMIT"):
                    _EMIT_LOG.append(("hmin", p, np.array(hv[idx_min], float), _hi))
                    _EMIT_LOG.append(("hmax", p, np.array(hv[idx_max], float), _hi))

        # ---- backward boundary scan (x-maxima): hybrid flat-center / arc-apex ----
        if ms:                   # SCC order: x-maxima on the same rotated boundary
            _mx = _cyclic_extrema_mask(np.round(nb_f[:, 0], _BWD_RND), find_min=False)
            subcritP = nb_f[np.flatnonzero(_mx)]
        else:
            subcritP = _backward_maxima(nb_b, nb_f)
        if _os.environ.get("OSCC_MCD_DBG"):
            xr = np.round(nb_b[:, 0], 0)
            xmax = xr.max()
            flat = nb_b[xr == xmax]
            print("  [MCD bwd p=%d] x-max=%.0f flat y-range [%.1f, %.1f] n=%d  bwd critP y=%s"
                  % (p, xmax, flat[:, 1].min(), flat[:, 1].max(), len(flat),
                     np.round(subcritP[:, 1], 1).tolist() if subcritP.size else []),
                  flush=True)
        critP = np.vstack([critP, subcritP])
        for pt in subcritP:
            cvex.append(_cvex_outer(inPoly, pt, find_min=False))
        if _os.environ.get("OSCC_MCD_EMIT"):
            _EMIT_LOG.append(("bwd", p, np.array(subcritP, float), None))

        cvex = np.array(cvex, dtype=int)

        # ---- decompose: vertical split-edges at concave critical points ----
        Vy = pw[p].Vertices[:, 1]
        Vy = Vy[~np.isnan(Vy)]
        ymin = round(float(np.min(Vy)))
        ymax = round(float(np.max(Vy)))
        for c in range(critP.shape[0]):
            in_arr = _scan_inside(pw[p], critP[c, 0], ymin, ymax)
            if in_arr.shape[0] != 0:
                if ymax - in_arr[-1, 1] < 50:
                    in_arr[-1, 1] = ymax
                if in_arr[0, 1] - ymin < 50:
                    in_arr[0, 1] = ymin
            if c < cvex.size and cvex[c] == 0:
                if not np.any(np.isnan(in_arr[:, 0])):
                    splitEdge.append(pu.polybuffer([in_arr[0], in_arr[-1]], _T, kind="lines"))
                else:
                    nanr = np.flatnonzero(np.isnan(in_arr[:, 1]))
                    in_arr[nanr - 1, 1] -= 1
                    in_arr[nanr + 1, 1] += 1
                    oo = in_arr[~np.isnan(in_arr[:, 0])]
                    bool_found = False
                    boo11 = boo12 = False
                    kfound = None
                    nseg = int(np.sum(np.isnan(in_arr[:, 0]))) + 1
                    for k in range(1, nseg + 1):
                        if oo[2 * k - 2, 1] < critP[c, 1] and oo[2 * k - 1, 1] > critP[c, 1]:
                            bool_found = True
                            boo11 = True
                            kfound = k
                            break
                    kk = None
                    if not bool_found:
                        sgn = (oo[:, 1] - critP[c, 1]) >= 0
                        d = np.flatnonzero(np.diff(sgn.astype(int)) != 0)
                        if d.size:
                            kk = [int(d[0]), int(d[0]) + 1]   # 0-based positions
                            bool_found = True
                            boo12 = True
                    if bool_found:
                        if boo11:
                            seg = np.array([oo[2 * kfound - 2] + [0, -10], oo[2 * kfound - 1] + [0, 10]])
                            splitEdge.append(pu.polybuffer(seg, _T, kind="lines"))
                        if boo12:
                            seg = np.array([oo[kk[0] - 1] + [0, -10], oo[kk[1] + 1] + [0, 10]])
                            splitEdge.append(pu.polybuffer(seg, _T, kind="lines"))

        # ---- merge critical points closer than 25 px (drop the concave duplicate) ----
        if critP.shape[0] > 1:
            n = critP.shape[0]
            distt = np.full((n, n), np.inf)
            for ii in range(n):
                d = spdist(critP[ii], critP)
                d[d == 0] = np.inf
                distt[:, ii] = d
            if np.min(distt) <= 25:
                A = (distt == np.min(distt))
                keep_rows = np.ones(n, dtype=bool)
                keep_rows[np.flatnonzero(cvex == 1)] = False
                A = A[keep_rows]
                drop = np.zeros(n, dtype=bool)
                for row in A:
                    drop |= row
                critP = critP[~drop]

        critPT = np.vstack([critPT, critP])

    # ---- apply split-edges, sort, return cells ----
    work = polyin_work if isinstance(polyin_work, PolyShape) else pu.union(polyin_work)
    base = polyin if isinstance(polyin, PolyShape) else pu.union(polyin)
    for se in splitEdge:
        work = pu.subtract(work, se)
        base = pu.subtract(base, se)
    work = pu.sortregions(work, "centroid", "ascend")
    base = pu.sortregions(base, "centroid", "ascend")
    polyout_work = pu.regions(work)
    polyout = pu.regions(base)
    return critPT, polyout_work, polyout, splitEdge


def _zero_nan_adjacent_1d(tf, values):
    tf = np.asarray(tf, dtype=bool).copy()
    nanrows = np.flatnonzero(np.isnan(values[:, 0]))
    if nanrows.size:
        idx = np.concatenate([nanrows + 1, nanrows - 1])
        idx = idx[(idx >= 0) & (idx < tf.size)]
        tf[idx] = False
    return tf

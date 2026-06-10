"""Polygon geometry utilities for the crack-filling coverage planner.

This module provides the core polygon operations used throughout the planning
pipeline: buffering/offsetting, Boolean operations (union, subtract, intersect),
point-in-polygon queries, and consistent vertex/boundary accessors.

All geometry is backed by Shapely, but the :class:`PolyShape` wrapper and its
accessors enforce a consistent vertex ordering that the downstream Morse Cell
Decomposition (MCD) and cell-traversal logic depend on:

- Outer boundary vertices stored **clockwise**; hole vertices **counter-clockwise**.
- Each ring starts at the **lexicographic-minimum** vertex (minimum x, then minimum y).
- :attr:`PolyShape.Vertices` — open rings (no closing repeat), multiple loops
  separated by ``[NaN, NaN]`` rows, outer ring first then its holes.
- :meth:`PolyShape.boundary` — closed rings (first vertex repeated at the end),
  ``[NaN, NaN]``-separated per loop.

Geometric shape fidelity (areas, buffer shapes, Boolean results) comes from
Shapely (or the Clipper library via ``pyclipper`` when available for higher
precision).
"""

import os as _os

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union

# Shapely buffer resolution: 180 arc vertices per full circle
# (quad_segs = segments per quarter circle; 4*45 = 180).
_QSEG = 45
_JOIN = {"round": 1, "mitre": 2, "miter": 2, "bevel": 3}
_CAP = {"round": 1, "flat": 2, "square": 3}


# --------------------------------------------------------------------------- #
# ring normalization
# --------------------------------------------------------------------------- #
def _ring_open(coords):
    """Return an open ring (no closing-duplicate vertex) as an (N, 2) float array."""
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[0] == 0:
        return c.reshape(-1, 2)
    if c.shape[0] > 1 and np.allclose(c[0], c[-1]):
        c = c[:-1]
    return c


def _signed_area(c):
    """Shoelace signed area of a ring; positive means counter-clockwise (math convention, y-up)."""
    x, y = c[:, 0], c[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _lexmin_index(c):
    """Return the index of the lexicographic-minimum vertex (minimum x, breaking ties by minimum y)."""
    return int(np.lexsort((c[:, 1], c[:, 0]))[0])


def _normalize_ring(coords, is_hole):
    """Reorder a ring to canonical convention: clockwise outer / counter-clockwise hole, starting at the lexicographic-minimum vertex."""
    c = _ring_open(coords)
    if c.shape[0] < 3:
        return c
    is_ccw = _signed_area(c) > 0
    if is_ccw != is_hole:          # outer wants CW (not ccw); hole wants CCW
        c = c[::-1]
    ix = _lexmin_index(c)
    return np.roll(c, -ix, axis=0)


def _iter_polys(geom):
    """Return component Polygons of a Polygon or MultiPolygon, sorted by ascending centroid distance from the origin.

    Empty or degenerate components are dropped. The ordering is deterministic
    across all platforms and matches the region ordering used by the planner's
    cell-decomposition logic.
    """
    if geom is None or geom.is_empty:
        return []
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    polys = [p for p in polys if p is not None and not p.is_empty and p.area > 0]
    if not polys:
        return []
    # Sort regions by ascending Euclidean distance from each region's centroid to the
    # origin. This produces a deterministic, geometry-consistent region order.
    polys.sort(key=lambda p: float(np.hypot(p.centroid.x, p.centroid.y)))
    return polys


def _parts(geom):
    """Return a list of normalized rings for all regions: each region's clockwise outer ring followed by its counter-clockwise hole rings."""
    parts = []
    for poly in _iter_polys(geom):
        parts.append(_normalize_ring(poly.exterior.coords, is_hole=False))
        for interior in poly.interiors:
            parts.append(_normalize_ring(interior.coords, is_hole=True))
    return parts


def _join_nan(parts, close):
    """Concatenate a list of rings into one (M, 2) array separated by ``[NaN, NaN]`` rows.

    Parameters
    ----------
    parts : list of (N_i, 2) arrays
        Individual boundary rings.
    close : bool
        If True, append the first vertex to each ring before joining.
    """
    if not parts:
        return np.empty((0, 2))
    chunks = []
    for i, p in enumerate(parts):
        if i > 0:
            chunks.append(np.array([[np.nan, np.nan]]))
        if close and p.shape[0] >= 1:
            p = np.vstack([p, p[:1]])
        chunks.append(p)
    return np.vstack(chunks)


# --------------------------------------------------------------------------- #
# PolyShape wrapper
# --------------------------------------------------------------------------- #
class PolyShape:
    """A polygon (or multi-polygon) with a consistent vertex-ordering convention.

    Wraps a Shapely Polygon or MultiPolygon and exposes two vertex views that
    the planning pipeline relies on for deterministic decomposition:

    - :attr:`Vertices` — open rings (no closing repeat), ``[NaN, NaN]``-separated,
      outer ring first then its holes.
    - :meth:`boundary` — closed rings (first vertex repeated at the end),
      ``[NaN, NaN]``-separated per loop.

    Both views order outer boundaries clockwise and hole boundaries
    counter-clockwise, starting each ring at its lexicographic-minimum vertex.
    """

    def __init__(self, geom=None):
        self.geom = geom

    @property
    def Vertices(self):
        """Open vertex array (N, 2), outer ring then holes, ``[NaN, NaN]``-separated.

        Returns
        -------
        numpy.ndarray, shape (M, 2)
            All ring vertices concatenated. Rings are clockwise (outer) or
            counter-clockwise (holes), starting at the lexicographic-minimum
            vertex, with no closing duplicate.
        """
        return _join_nan(_parts(self.geom), close=False)

    def boundary(self):
        """Closed boundary array (M, 2), outer ring then holes, ``[NaN, NaN]``-separated.

        Returns
        -------
        numpy.ndarray, shape (M, 2)
            Same as :attr:`Vertices` but each ring is closed (first vertex repeated
            at the end).
        """
        return _join_nan(_parts(self.geom), close=True)

    @property
    def area(self):
        return 0.0 if (self.geom is None or self.geom.is_empty) else float(self.geom.area)

    @property
    def is_empty(self):
        return self.geom is None or self.geom.is_empty

    def __repr__(self):
        n = 0 if self.is_empty else len(self.Vertices)
        return f"PolyShape(area={self.area:.3f}, nverts={n})"


# --------------------------------------------------------------------------- #
# constructors / basic ops
# --------------------------------------------------------------------------- #
def polyshape(x=None, y=None):
    """Construct a :class:`PolyShape` from vertex coordinates.

    Parameters
    ----------
    x : array-like or (N, 2) array-like
        X-coordinates, or a full (N, 2) coordinate array when ``y`` is omitted.
        Separate boundary loops may be delimited by ``[NaN, NaN]`` rows.
    y : array-like, optional
        Y-coordinates (same length as ``x``). Holes are auto-detected by containment.

    Returns
    -------
    PolyShape
    """
    if x is None:
        return PolyShape(None)
    if y is None:
        coords = np.asarray(x, dtype=float)
    else:
        coords = np.column_stack([np.asarray(x, dtype=float).ravel(),
                                  np.asarray(y, dtype=float).ravel()])
    return PolyShape(_geom_from_loops(coords))


def _split_nan(coords):
    """Split an (N, 2) array on ``NaN`` rows into a list of (M_i, 2) loop arrays."""
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return []
    nan_rows = np.isnan(coords).any(axis=1)
    loops, cur = [], []
    for i in range(coords.shape[0]):
        if nan_rows[i]:
            if cur:
                loops.append(np.asarray(cur)); cur = []
        else:
            cur.append(coords[i])
    if cur:
        loops.append(np.asarray(cur))
    return loops


def _geom_from_loops(coords):
    """Build a Shapely Polygon or MultiPolygon from ``NaN``-separated vertex loops.

    Outer rings and holes are distinguished by containment: a ring is classified
    as a hole if a strictly larger ring fully covers it.
    """
    loops = [_ring_open(l) for l in _split_nan(coords) if _ring_open(l).shape[0] >= 3]
    if not loops:
        return None
    rings = [Polygon(l) for l in loops]
    # classify: a ring is a hole if a strictly-larger ring covers it. Test
    # WHOLE-ring containment, not a representative point -- an outer ring's
    # representative point can land inside one of its own holes (which would
    # mis-flag the outer ring as contained and drop the whole region).
    shells, holes = [], []
    for i, ri in enumerate(rings):
        contained = any(j != i and rings[j].area > ri.area and rings[j].covers(ri)
                        for j in range(len(rings)))
        (holes if contained else shells).append(loops[i])
    polys = []
    for s in shells:
        sp = Polygon(s)
        sh = [h for h in holes if sp.area > Polygon(h).area and sp.covers(Polygon(h))]
        polys.append(Polygon(s, sh))
    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    return geom.buffer(0) if not geom.is_valid else geom


def regions(ps):
    """Return a list of single-region :class:`PolyShape` objects from a (possibly multi-region) PolyShape.

    Regions are ordered by ascending centroid distance from the origin.

    Parameters
    ----------
    ps : PolyShape

    Returns
    -------
    list of PolyShape
    """
    if ps.is_empty:
        return []
    return [PolyShape(p) for p in _iter_polys(ps.geom)]


def holes(ps):
    """Return each interior hole of ``ps`` as a filled :class:`PolyShape`.

    Parameters
    ----------
    ps : PolyShape

    Returns
    -------
    list of PolyShape
    """
    out = []
    if ps.is_empty:
        return out
    for poly in _iter_polys(ps.geom):
        for interior in poly.interiors:
            out.append(PolyShape(Polygon(interior)))
    return out


def rmholes(ps):
    """Return a copy of ``ps`` with all interior holes removed.

    Parameters
    ----------
    ps : PolyShape

    Returns
    -------
    PolyShape
    """
    if ps.is_empty:
        return PolyShape(None)
    polys = [Polygon(p.exterior) for p in _iter_polys(ps.geom)]
    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    return PolyShape(geom)


def area(ps):
    return ps.area


# --------------------------------------------------------------------------- #
# buffers / boolean ops / queries
# --------------------------------------------------------------------------- #
def _as_geom(x):
    return x.geom if isinstance(x, PolyShape) else x


# Circular disk: 180-gon sampled at 2-degree steps (CW), exact radius.
# The start angle matters: for a 'points' disk the start is -91 deg; for a
# 'lines' buffer the cap/join arcs are edge-aligned (start = the adjacent
# edge's outward-normal angle).  Using this consistent arc grid ensures
# pathfinder/line_of_sight_poly route over the buffer boundary correctly.
_POINTS_DISK_ANG = np.radians(-91.0 - 2.0 * np.arange(180))
_POINTS_DISK_UNIT = np.column_stack([np.cos(_POINTS_DISK_ANG), np.sin(_POINTS_DISK_ANG)])


def _disk(center, r, start_deg=None):
    """Build a 180-vertex circular disk polygon of radius ``r`` centered at ``center``.

    The arc is sampled clockwise at 2-degree steps.  ``start_deg`` controls the
    angle of the first vertex; if omitted the default start angle (-91°) is used.
    """
    if start_deg is None:
        unit = _POINTS_DISK_UNIT
    else:
        th = np.radians(start_deg - 2.0 * np.arange(180))
        unit = np.column_stack([np.cos(th), np.sin(th)])
    return Polygon(unit * r + np.asarray(center, dtype=float))


_STEP = np.radians(2.0)   # arc step: 2 deg (180 vertices per full circle)


def _arc_excl(c, r, a0, a1):
    """Return interior arc vertices (endpoints excluded) in clockwise order.

    The arc runs clockwise from angle ``a0`` to ``a1`` on a fixed 2-degree grid
    anchored at the end angle ``a1``, matching the arc sampling used for the
    line-buffer cap/join construction.
    """
    d = (a0 - a1) % (2 * np.pi)        # positive CW span
    n = int(round(d / _STEP))
    out = []
    for k in range(n - 1, 0, -1):
        th = a1 + k * _STEP
        out.append(c + r * np.array([np.cos(th), np.sin(th)]))
    return out


def _line_isect(p0, d0, p1, d1):
    """Return the intersection point of lines ``p0 + t*d0`` and ``p1 + s*d1``."""
    A = np.array([[d0[0], -d1[0]], [d0[1], -d1[1]]])
    return p0 + np.linalg.solve(A, p1 - p0)[0] * d0


def _line_buffer(pts, r):
    """Build the offset polygon of a polyline with radius ``r``.

    Constructs a single clockwise outer ring using an ordered offset-curve:
    a forward pass along the left (+normal) side, a 180-degree semicircular end
    cap, a backward pass along the right (-normal) side, and a 180-degree start
    cap.  Convex joins and caps are rounded with 2-degree arc steps; concave
    joins are handled with a sharp mitre (offset-line intersection); collinear
    interior vertices are absorbed.

    Falls back to the Clipper open-round buffer for self-overlapping input paths
    (e.g. a tight zig-zag) where the ordered offset curve would self-intersect.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    # drop consecutive duplicate vertices
    keep = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - keep[-1])) > 1e-9:
            keep.append(p)
    pts = np.array(keep)
    N = len(pts)
    if N == 1:
        return _disk(pts[0], r)
    # The ordered offset curve below is only well-defined for a SIMPLE
    # (non-self-intersecting) polyline. A self-overlapping input path makes
    # concave mitres spike to ±Inf, so those are routed to the Clipper
    # open-round buffer instead.
    if not LineString(pts).is_simple:
        return _line_buffer_fallback(pts, r)
    u = [(pts[i + 1] - pts[i]) / np.hypot(*(pts[i + 1] - pts[i])) for i in range(N - 1)]
    nl = [np.array([-ui[1], ui[0]]) for ui in u]          # left normal per segment
    ang = lambda v: np.arctan2(v[1], v[0])
    TOL = 1e-9

    def cross(i):   # turn sign at interior vertex i (edge i-1 -> edge i)
        return u[i - 1][0] * u[i][1] - u[i - 1][1] * u[i][0]

    seq = []
    # forward pass, +normal (left) side
    for i in range(N - 1):
        if i == 0:
            seq.append(pts[i] + r * nl[i])
        else:
            c = cross(i)
            if abs(c) < TOL:                 # collinear -> drop redundant vertex
                seq.pop()
            elif c < 0:                      # right turn -> convex on +side -> arc
                seq += _arc_excl(pts[i], r, ang(nl[i - 1]), ang(nl[i]))
                seq.append(pts[i] + r * nl[i])
            else:                            # left turn -> concave -> mitre
                seq.pop()
                seq.append(_line_isect(pts[i] + r * nl[i - 1], u[i - 1],
                                       pts[i] + r * nl[i], u[i]))
        seq.append(pts[i + 1] + r * nl[i])
    # end cap at pts[-1] (180 deg)
    seq += _arc_excl(pts[-1], r, ang(nl[N - 2]), ang(-nl[N - 2]))
    # backward pass, -normal side
    for i in range(N - 2, -1, -1):
        if i == N - 2:
            seq.append(pts[i + 1] - r * nl[i])
        else:
            j = i + 1
            c = cross(j)
            if abs(c) < TOL:
                seq.pop()
            elif c > 0:                      # left turn -> convex on -side -> arc
                seq += _arc_excl(pts[j], r, ang(-nl[j]), ang(-nl[i]))
                seq.append(pts[i + 1] - r * nl[i])
            else:                            # concave -> mitre
                seq.pop()
                seq.append(_line_isect(pts[j] - r * nl[j], u[j],
                                       pts[j] - r * nl[i], u[i]))
        seq.append(pts[i] - r * nl[i])
    # start cap at pts[0] (180 deg)
    seq += _arc_excl(pts[0], r, ang(-nl[0]), ang(nl[0]))

    if np.isfinite(np.asarray(seq, dtype=float)).all():
        poly = Polygon(seq)
        if poly.is_valid and not poly.is_empty and _all_finite(poly):
            return poly
    # The ordered offset curve self-intersected or went non-finite (e.g. a tight
    # coverage zig-zag where lane spacing < r causes inner offsets to cross).
    # Fall back to the Clipper open-round buffer, which correctly unions the
    # full swept region.
    return _line_buffer_fallback(pts, r)


def polybuffer(obj, r, kind=None, joint="round", miter_limit=4.0):
    """Buffer or offset a geometry by distance ``r``.

    Three calling modes:

    - ``polybuffer(pts, r, kind='lines')`` — rounded stadium buffer around a
      polyline (round caps and joins, 2-degree arc steps).
    - ``polybuffer(pts, r, kind='points')`` — union of disks of radius ``r``
      centered at each point in ``pts``.
    - ``polybuffer(poly, r)`` — offset a :class:`PolyShape` outward by ``r``
      (negative ``r`` shrinks the polygon inward).

    Parameters
    ----------
    obj : array-like or PolyShape
        Polyline points, point set, or polygon to buffer.
    r : float
        Buffer radius (pixels).  Negative values shrink a polygon.
    kind : {'lines', 'points'} or None
        Buffer mode for point/polyline inputs.  ``None`` selects polygon offset.
    joint : str, optional
        Join style for polygon offset (``'round'``, ``'mitre'``, ``'bevel'``).
    miter_limit : float, optional
        Miter-limit factor for mitre joins.

    Returns
    -------
    PolyShape
    """
    if kind in ("lines", "line"):
        return PolyShape(_line_buffer(obj, r))
    if kind == "points":
        pts = np.atleast_2d(np.asarray(obj, dtype=float))
        return PolyShape(unary_union([_disk(p, r) for p in pts]))
    # polygon offset
    g = _as_geom(obj)
    if g is None or g.is_empty:
        # Offsetting an empty region returns empty; guard so the MCD-prep
        # buffer doesn't crash at end-of-coverage.
        return PolyShape(None)
    # Prefer the Clipper library (pyclipper) for polygon offset: it produces
    # ~180 arc vertices per full circle and <0.12px shape error at the working
    # coordinate scale. Falls back to Shapely if pyclipper is unavailable or
    # OSCC_NO_CLIPPER is set.
    if not _os.environ.get("OSCC_NO_CLIPPER"):
        res = _clipper_offset(g, float(r))
        if res is not None:
            return res
    out = g.buffer(r, quad_segs=_QSEG, join_style=_JOIN.get(joint, 1),
                   mitre_limit=miter_limit)
    return PolyShape(out)


# Clipper integer scale: 1e5 gives <0.12px offset error on the crack-map
# coordinate range while keeping Clipper's int64 area products two orders
# below the int64 ceiling.  Do not raise this value — it perturbs the offset
# operations without improving accuracy at this coordinate scale.
_CLIPPER_SCALE = 1e5


def _clipper_offset(g, r):
    """Offset a Shapely polygon by ``r`` using the Clipper library (round joins, 180 vertices per full circle).

    Returns a :class:`PolyShape`, or ``None`` if ``pyclipper`` is unavailable
    or yields an empty result (the caller falls back to Shapely).
    """
    try:
        import pyclipper
    except ImportError:
        return None
    polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    arc_tol = abs(r) * _CLIPPER_SCALE * (1.0 - np.cos(np.pi / 180.0))
    pco = pyclipper.PyclipperOffset(miter_limit=3.0, arc_tolerance=max(arc_tol, 1e-9))
    for poly in polys:
        pco.AddPath(pyclipper.scale_to_clipper(list(poly.exterior.coords)[:-1], _CLIPPER_SCALE),
                    pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        for ring in poly.interiors:
            pco.AddPath(pyclipper.scale_to_clipper(list(ring.coords)[:-1], _CLIPPER_SCALE),
                        pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    sol = pco.Execute(r * _CLIPPER_SCALE)
    if not sol:
        return PolyShape(None)
    loops = []
    for i, s in enumerate(sol):
        if i > 0:
            loops.append(np.array([[np.nan, np.nan]]))
        loops.append(np.array(pyclipper.scale_from_clipper(s, _CLIPPER_SCALE), dtype=float))
    return PolyShape(_geom_from_loops(np.vstack(loops)))


def _clipper_line_buffer(pts, r):
    """Open-polyline buffer via Clipper (round joins and round caps).

    Used as the fallback for self-overlapping polylines (e.g. tight zig-zag
    coverage paths) where the ordered offset curve would self-intersect.
    Clipper's open-round buffer correctly unions the full swept region.

    Returns a :class:`PolyShape`, or ``None`` if ``pyclipper`` is unavailable
    or yields an empty result (the caller falls back to Shapely).
    """
    try:
        import pyclipper
    except ImportError:
        return None
    pts = np.atleast_2d(np.asarray(pts, dtype=float))[:, :2]
    arc_tol = abs(r) * _CLIPPER_SCALE * (1.0 - np.cos(np.pi / 180.0))
    pco = pyclipper.PyclipperOffset(miter_limit=3.0, arc_tolerance=max(arc_tol, 1e-9))
    pco.AddPath(pyclipper.scale_to_clipper(pts.tolist(), _CLIPPER_SCALE),
                pyclipper.JT_ROUND, pyclipper.ET_OPENROUND)
    sol = pco.Execute(r * _CLIPPER_SCALE)
    if not sol:
        return PolyShape(None)
    loops = []
    for i, s in enumerate(sol):
        if i > 0:
            loops.append(np.array([[np.nan, np.nan]]))
        loops.append(np.array(pyclipper.scale_from_clipper(s, _CLIPPER_SCALE), dtype=float))
    return PolyShape(_geom_from_loops(np.vstack(loops)))


def _line_buffer_fallback(pts, r):
    """Fallback 'lines' buffer for self-overlapping polylines.

    Tries the Clipper open-round buffer first (preferred for accuracy);
    falls back to Shapely if ``pyclipper`` is unavailable or the environment
    variable ``OSCC_NO_CLIPPER`` is set.
    """
    if not _os.environ.get("OSCC_NO_CLIPPER"):
        cl = _clipper_line_buffer(pts, r)
        if cl is not None and not cl.is_empty:
            return _as_geom(cl)
    return LineString(np.atleast_2d(pts)[:, :2]).buffer(
        r, quad_segs=_QSEG, cap_style=1, join_style=1)


def subtract(A, B, keep_collinear=False):
    """Return the polygon difference ``A \\ B`` (region of ``A`` not covered by ``B``).

    If ``B`` is empty, ``A`` is returned unchanged.

    When the environment variable ``OSCC_CLIPPER_SUBTRACT`` is set, the Boolean
    difference is computed via the Clipper library (using ``pyclipper``) for
    higher geometric precision, which can produce a more deterministic free-space
    region and boundary ordering.  The default uses Shapely's difference engine.

    Parameters
    ----------
    A, B : PolyShape or Shapely geometry

    Returns
    -------
    PolyShape
    """
    ga, gb = _as_geom(A), _as_geom(B)
    if gb is None or gb.is_empty:
        return PolyShape(ga)
    if _os.environ.get("OSCC_CLIPPER_SUBTRACT"):
        res = _clipper_subtract(ga, gb)
        if res is not None:
            return PolyShape(res)
    return PolyShape(ga.difference(gb))


def _clipper_subtract(ga, gb):
    """Compute the polygon difference ``A \\ B`` using Clipper's DIFFERENCE operation.

    Returns a Shapely geometry, or ``None`` if ``pyclipper`` is unavailable.
    """
    try:
        import pyclipper
    except ImportError:
        return None
    from shapely.geometry import Polygon, MultiPolygon
    sc = _CLIPPER_SCALE
    pc = pyclipper.Pyclipper()

    def add(geom, pt):
        for poly in _iter_polys(geom):
            pc.AddPath(pyclipper.scale_to_clipper(list(poly.exterior.coords)[:-1], sc), pt, True)
            for ring in poly.interiors:
                pc.AddPath(pyclipper.scale_to_clipper(list(ring.coords)[:-1], sc), pt, True)
    add(ga, pyclipper.PT_SUBJECT)
    add(gb, pyclipper.PT_CLIP)
    tree = pc.Execute2(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

    polys = []

    def walk(node):
        for child in node.Childs:
            if not child.IsHole:
                ext = pyclipper.scale_from_clipper(child.Contour, sc)
                holes = [pyclipper.scale_from_clipper(h.Contour, sc)
                         for h in child.Childs if h.IsHole]
                if len(ext) >= 3:
                    polys.append(Polygon(ext, [h for h in holes if len(h) >= 3]))
                for h in child.Childs:          # nested outers (islands inside holes)
                    walk(h)
    walk(tree)
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def regCombine(ps, r=1.0):
    """Close small gaps in a polygon by dilating, unioning, then eroding.

    Applies ``polybuffer(union(polybuffer(ps, r)), -r)``.  Useful for merging
    regions that were split by narrow cut lines (e.g. cell-decomposition bars).

    Parameters
    ----------
    ps : PolyShape
    r : float, optional
        Dilation/erosion radius (default 1.0 pixel).

    Returns
    -------
    PolyShape
    """
    grown = polybuffer(ps, r)
    return polybuffer(union(grown), -r)


def rmslivers(ps, tol=1e-5):
    """Remove boundary slivers and near-coincident vertices from a polygon.

    Deduplicates near-coincident vertices and removes near-zero-area spikes
    (antenna artifacts) from each boundary ring.  Boundary outliers that arise
    from repeated buffer round-trips at concave pinch points are cleaned up,
    keeping the vertex count and boundary shape compact.

    Uses the Clipper ``CleanPolygon`` operation when ``pyclipper`` is available
    (and ``OSCC_NO_CLIPPER`` is not set); otherwise falls back to a Shapely
    ``buffer(0)`` validity repair.

    Parameters
    ----------
    ps : PolyShape
    tol : float, optional
        Cleaning distance threshold (default ``1e-5``).

    Returns
    -------
    PolyShape
    """
    g = _as_geom(ps)
    if g is None or g.is_empty:
        return PolyShape(g)
    if not _os.environ.get("OSCC_NO_CLIPPER"):
        res = _clipper_clean(g, float(tol))
        if res is not None:
            return res
    return PolyShape(g.buffer(0))


def _clipper_clean(g, tol):
    """Clean each boundary ring of ``g`` using Clipper's ``CleanPolygon`` operation.

    Returns a :class:`PolyShape`, or ``None`` if ``pyclipper`` is unavailable
    (the caller falls back to a Shapely ``buffer(0)``).
    """
    try:
        import pyclipper
    except ImportError:
        return None
    dist = max(tol * _CLIPPER_SCALE, 0.0)
    polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]

    def clean_ring(coords):
        c = list(coords)
        if len(c) >= 2 and c[0] == c[-1]:
            c = c[:-1]                      # CleanPolygon wants an open ring
        if len(c) < 3:
            return None
        scaled = pyclipper.scale_to_clipper(c, _CLIPPER_SCALE)
        out = pyclipper.CleanPolygon(scaled, dist)
        if not out or len(out) < 3:
            return None
        return np.array(pyclipper.scale_from_clipper(out, _CLIPPER_SCALE), dtype=float)

    loops = []
    for poly in polys:
        ext = clean_ring(poly.exterior.coords)
        if ext is None:
            continue
        if loops:
            loops.append(np.array([[np.nan, np.nan]]))
        loops.append(ext)
        for ring in poly.interiors:
            hole = clean_ring(ring.coords)
            if hole is not None:
                loops.append(np.array([[np.nan, np.nan]]))
                loops.append(hole)
    if not loops:
        return PolyShape(None)
    return PolyShape(_geom_from_loops(np.vstack(loops)))


def _coords_finite(coords):
    """Return finite (non-NaN, non-Inf) vertices of a ring, or ``None`` if fewer than 3 remain."""
    c = _ring_open(coords)
    c = c[np.isfinite(c).all(axis=1)]
    return c if c.shape[0] >= 3 else None


def _all_finite(geom):
    """Return ``True`` if every vertex in every ring of ``geom`` is finite (no NaN or Inf)."""
    for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom]):
        if not np.isfinite(np.asarray(poly.exterior.coords)).all():
            return False
        for ring in poly.interiors:
            if not np.isfinite(np.asarray(ring.coords)).all():
                return False
    return True


def sanitize(ps):
    """Remove non-finite vertices and return a well-formed :class:`PolyShape`.

    Drops any NaN or Inf coordinates from each ring and revalidates the
    resulting geometry.  The buffer-and-subtract pipeline can occasionally
    produce a non-finite spike at a near-degenerate concave mitre; those
    vertices cause downstream Boolean operations to raise geometry exceptions.
    This function rebuilds the polygon from finite vertices only, making it
    safe to pass to union, subtract, and intersection operations.  Returns the
    polygon unchanged when it is already finite and valid.

    Parameters
    ----------
    ps : PolyShape or Shapely geometry

    Returns
    -------
    PolyShape
    """
    g = _as_geom(ps)
    if g is None or g.is_empty:
        return PolyShape(g)
    if _all_finite(g) and g.is_valid:
        return PolyShape(g)
    polys = []
    for poly in (g.geoms if isinstance(g, MultiPolygon) else [g]):
        shell = _coords_finite(poly.exterior.coords)
        if shell is None:
            continue
        rings = [h for h in (_coords_finite(i.coords) for i in poly.interiors)
                 if h is not None]
        p = Polygon(shell, rings)
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty and p.area > 0:
            polys.append(p)
    if not polys:
        return PolyShape(None)
    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return PolyShape(geom)


def union(*args):
    """Return the geometric union of a collection of :class:`PolyShape` objects.

    Accepts either a single list/tuple of PolyShapes or multiple PolyShape
    arguments.

    Parameters
    ----------
    *args : PolyShape or list of PolyShape

    Returns
    -------
    PolyShape
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
        geoms = [_as_geom(p) for p in args[0]]
    else:
        geoms = [_as_geom(p) for p in args]
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return PolyShape(None)
    return PolyShape(unary_union(geoms))


def intersect(A, B):
    """Return the geometric intersection of two :class:`PolyShape` objects.

    Parameters
    ----------
    A, B : PolyShape or Shapely geometry

    Returns
    -------
    PolyShape
    """
    return PolyShape(_as_geom(A).intersection(_as_geom(B)))


def intersect_line(ps, seg):
    """Clip a line segment against a polygon and return the inside and outside parts.

    Parameters
    ----------
    ps : PolyShape
        The clipping polygon.
    seg : array-like, shape (K, 2)
        Polyline or segment to clip.

    Returns
    -------
    inside : Shapely geometry
        The part of ``seg`` that lies inside ``ps``.
    outside : Shapely geometry
        The part of ``seg`` that lies outside ``ps``.

    Notes
    -----
    Both returned geometries are Shapely ``LineString`` / ``MultiLineString`` or
    empty.  Callers can test ``.is_empty`` and ``.length``.
    """
    g = _as_geom(ps)
    line = LineString(np.atleast_2d(np.asarray(seg, dtype=float)))
    if g is None or g.is_empty:
        return LineString([]), line
    return line.intersection(g), line.difference(g)


def _ordered_clip(line, geom):
    """Clip a Shapely line against a geometry and return endpoint pairs ordered along the line.

    Each disjoint inside piece becomes a two-row ``[entry; exit]`` pair.  Pieces
    are ordered by their position along the original line and separated by
    ``[NaN, NaN]`` rows in the output array.
    """
    if geom is None or geom.is_empty:
        return np.empty((0, 2))
    if geom.geom_type == "LineString":
        comps = [geom]
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        comps = [c for c in geom.geoms if c.geom_type == "LineString" and not c.is_empty]
    else:
        comps = []
    segs = []
    for c in comps:
        coords = np.asarray(c.coords, dtype=float)
        params = np.array([line.project(Point(p)) for p in coords])
        order = np.argsort(params)
        coords = coords[order]
        segs.append((params[order][0], coords[[0, -1]]))
    segs.sort(key=lambda s: s[0])
    out = []
    for i, (_, pair) in enumerate(segs):
        if i > 0:
            out.append(np.array([[np.nan, np.nan]]))
        out.append(pair)
    return np.vstack(out) if out else np.empty((0, 2))


def intersect_segment(ps, seg):
    """Return the inside and outside parts of a segment clipped against a polygon, as ordered arrays.

    Used by the Morse Cell Decomposition (MCD) vertical scan to split a vertical
    line segment into the portions inside and outside each decomposition cell.

    Parameters
    ----------
    ps : PolyShape
        The clipping polygon.
    seg : array-like, shape (K, 2)
        Segment or polyline to clip.

    Returns
    -------
    in_arr : numpy.ndarray, shape (M, 2)
        Inside pieces as ordered ``[entry; exit]`` pairs, ``[NaN, NaN]``-separated.
    out_arr : numpy.ndarray, shape (M, 2)
        Outside pieces, same format.
    """
    inside, outside = intersect_line(ps, seg)
    line = LineString(np.atleast_2d(np.asarray(seg, dtype=float)))
    return _ordered_clip(line, inside), _ordered_clip(line, outside)


def _close_loops_xy(xv, yv):
    """Close each NaN-separated loop by appending its start vertex if not already closed.

    Returns flat ``(xv, yv)`` 1-D arrays with single ``NaN`` separators between loops.
    """
    coords = np.column_stack([np.asarray(xv, float).ravel(),
                              np.asarray(yv, float).ravel()])
    loops = _split_nan(coords)
    out = []
    for k, loop in enumerate(loops):
        if k > 0:
            out.append(np.array([[np.nan, np.nan]]))
        if loop.shape[0] >= 1 and not np.array_equal(loop[0], loop[-1]):
            loop = np.vstack([loop, loop[0]])
        out.append(loop)
    if not out:
        return np.empty(0), np.empty(0)
    c = np.vstack(out)
    return c[:, 0], c[:, 1]


def _inpoly_winding(xq, yq, xv, yv, tol):
    """Hormann-Agathos (2001) quadrant winding number point-in-polygon test.

    Parameters
    ----------
    xq, yq : array-like
        Query point coordinates.
    xv, yv : array-like
        Closed, ``NaN``-separated polygon boundary vertices.
    tol : float or array-like
        Edge-closeness tolerance.  A scalar applies globally (used by
        :func:`isinterior`); an array of length ``Nv-1`` applies per edge
        (used by :func:`inpolygon`).

    Returns
    -------
    IN : numpy.ndarray of bool
        ``True`` for points inside or on the boundary.
    ON : numpy.ndarray of bool
        ``True`` for points that lie on a boundary edge or vertex.
    """
    xq = np.atleast_1d(np.asarray(xq, float)).ravel()
    yq = np.atleast_1d(np.asarray(yq, float)).ravel()
    M = xq.size
    xv = np.asarray(xv, float).ravel()
    yv = np.asarray(yv, float).ravel()
    Nv = xv.size
    if M == 0 or Nv < 2:
        return np.zeros(M, bool), np.zeros(M, bool)

    # translate every vertex to each query origin -> (Nv, M)
    xvt = xv[:, None] - xq[None, :]
    yvt = yv[:, None] - yq[None, :]
    posX = xvt > 0
    posY = yvt > 0
    # Quadrant convention: x==0 is treated as the negative side (negX = ~(x>0)).
    quad = ((~posX) & posY) * 1.0 + ((~posX) & (~posY)) * 2.0 + (posX & (~posY)) * 3.0
    quad[np.isnan(xvt) | np.isnan(yvt)] = np.nan

    m, mp1 = slice(0, Nv - 1), slice(1, Nv)
    cross = xvt[m] * yvt[mp1] - xvt[mp1] * yvt[m]      # (Nv-1, M)
    dot = xvt[m] * xvt[mp1] + yvt[m] * yvt[mp1]
    sgn = np.sign(cross)
    tol_arr = np.asarray(tol, float)
    below = np.abs(cross) < (tol_arr if tol_arr.ndim == 0 else tol_arr[:, None])
    sgn = np.where(below, 0.0, sgn)

    dq = np.diff(quad, axis=0)                          # (Nv-1, M)
    dq = np.where(np.abs(dq) == 3, -dq / 3.0, dq)       # wraparound Q0<->Q3
    a2 = np.abs(dq) == 2                                # ambiguous: use cross sign
    dq = np.where(a2, 2.0 * sgn, dq)
    dq = np.where(np.isnan(dq), 0.0, dq)               # zero NaN-separator steps

    winding = np.sum(dq, axis=0)
    ON = np.any((sgn == 0) & (dot <= 0), axis=0)
    IN = (winding != 0) | ON
    return IN, ON


def isinterior(ps, pts):
    """Test whether points lie inside or on the boundary of a polygon.

    Uses a winding-number algorithm with a global tolerance derived from the
    polygon's bounding box (``(max_abs_coord + max_extent) * 1e-12``).

    Parameters
    ----------
    ps : PolyShape
        The query polygon.
    pts : array-like, shape (N, 2)
        Points to test.

    Returns
    -------
    numpy.ndarray of bool, shape (N,)
        ``True`` for each point that is inside or on the boundary of ``ps``.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    g = _as_geom(ps)
    if g is None or g.is_empty or pts.size == 0:
        return np.zeros(pts.shape[0], dtype=bool)
    bnd = ps.boundary() if isinstance(ps, PolyShape) else PolyShape(g).boundary()
    minx, miny, maxx, maxy = g.bounds
    maxD = max(abs(minx), abs(maxx), abs(miny), abs(maxy))
    maxW = max(maxx - minx, maxy - miny)
    tol = (maxD + maxW) * 1e-12
    IN, _ = _inpoly_winding(pts[:, 0], pts[:, 1], bnd[:, 0], bnd[:, 1], tol)
    return IN


def inpolygon(xq, yq, xv, yv):
    """Test which query points lie inside or on the boundary of a polygon.

    Uses a Hormann-Agathos quadrant winding number with a per-edge tolerance
    ``scaledEps = max(|mid_x|, |mid_y|, |mid_x * mid_y|) * eps * 3`` based on
    each edge midpoint coordinate magnitude, matching the behavior of the
    underlying polygon library's ``inpolygon`` function.

    Parameters
    ----------
    xq, yq : array-like
        Query point coordinates.
    xv, yv : array-like
        Closed polygon boundary vertices (``NaN``-separated loops are supported).

    Returns
    -------
    IN : numpy.ndarray of bool
        ``True`` for points inside or on the boundary.
    ON : numpy.ndarray of bool
        ``True`` for points on a boundary edge or vertex.
    """
    xvc, yvc = _close_loops_xy(xv, yv)
    if xvc.size < 2:
        n = np.atleast_1d(np.asarray(xq, float)).ravel().size
        return np.zeros(n, bool), np.zeros(n, bool)
    avx = np.abs(0.5 * (xvc[:-1] + xvc[1:]))           # edge-midpoint |coords|
    avy = np.abs(0.5 * (yvc[:-1] + yvc[1:]))
    scale = np.maximum(np.maximum(avx, avy), avx * avy)
    scaled_eps = scale * np.finfo(float).eps * 3.0
    return _inpoly_winding(xq, yq, xvc, yvc, scaled_eps)


def sortregions(ps, by="centroid", order="ascend"):
    """Return a :class:`PolyShape` whose regions are sorted by centroid distance from the origin.

    Parameters
    ----------
    ps : PolyShape
    by : str, optional
        Sort key — currently only ``'centroid'`` is supported.
    order : {'ascend', 'descend'}, optional
        Sort direction (default ``'ascend'``).

    Returns
    -------
    PolyShape
    """
    polys = _iter_polys(ps.geom)  # already centroid-ascending
    if order == "descend":
        polys = polys[::-1]
    if not polys:
        return PolyShape(None)
    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    return PolyShape(geom)


def addboundary(ps, x, y):
    """Add a boundary loop to a :class:`PolyShape` by taking the union with the new loop.

    Parameters
    ----------
    ps : PolyShape
        Existing polygon.
    x, y : array-like
        Vertex coordinates of the new boundary loop to add.

    Returns
    -------
    PolyShape
    """
    new = Polygon(np.column_stack([np.asarray(x, float).ravel(),
                                   np.asarray(y, float).ravel()]))
    if ps.is_empty:
        return PolyShape(new)
    return PolyShape(unary_union([ps.geom, new]))

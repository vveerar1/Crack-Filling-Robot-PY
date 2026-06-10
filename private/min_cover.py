"""Minimum-length crack-covering path via taut-string relaxation.

Given a crack centerline and the robot's circular footprint radius ``a``, this
module computes a short path such that every point on the crack centerline lies
within distance ``a`` of some point on the path.  In other words, the robot's
footprint sweeps the entire crack when it traverses the path.

**Algorithm**

The crack centerline is first resampled at a uniform arc-length step of
approximately ``a / 3`` to produce an ordered sequence of disk centres
``c_0, c_1, ..., c_n``.  One path waypoint ``p_i`` is associated with each
disk centre, constrained to lie within a disk of radius ``r ≤ a`` centred at
``c_i``.  Starting with the waypoints placed on the crack itself (feasible
initial solution), the algorithm iterates a Gauss-Seidel relaxation: each
interior waypoint is moved toward the midpoint of its two neighbours
(straightening the path), then projected back onto the boundary of its disk if
it would otherwise escape.  The loop terminates when the maximum per-step
displacement falls below 0.05 pixels or after a fixed iteration count.
A Ramer-Douglas-Peucker simplification pass is applied at the end to remove
collinear redundant points.

The result converges to the *taut string* (rubber-band) solution, which is
provably minimum-length among all paths satisfying the coverage constraint.

**Public functions**

``min_cover_path(centerline, a, ...)``
    Fixed disk-fraction taut-string path.

``min_cover_path_adaptive(centerline, a, ...)``
    Adaptive version: binary-searches for the largest disk fraction whose path
    still satisfies ``max_gap(path, centerline) ≤ a``, recovering path-length
    savings that the fixed safety margin leaves behind.

``max_gap(centerline, path)``
    Maximum distance from any centerline point to the nearest point on the
    path (coverage verification metric).

``length(path)``
    Arc length of a polyline.
"""
import numpy as np


def _resample(poly, step):
    poly = np.asarray(poly, float)
    if len(poly) < 2:
        return poly
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return poly[[0, -1]]
    u = np.arange(0, s[-1], step)
    u = np.append(u, s[-1])
    x = np.interp(u, s, poly[:, 0]); y = np.interp(u, s, poly[:, 1])
    return np.column_stack([x, y])


def _dp(pts, tol):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool); keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        ab = pts[e] - pts[s]; L = np.hypot(*ab)
        if L < 1e-9:
            d = np.hypot(*(pts[s + 1:e] - pts[s]).T)
        else:
            d = np.abs(ab[0] * (pts[s + 1:e, 1] - pts[s, 1]) - ab[1] * (pts[s + 1:e, 0] - pts[s, 0])) / L
        k = int(np.argmax(d)) + s + 1
        if d.max() > tol:
            keep[k] = True; stack += [(s, k), (k, e)]
    return pts[keep]


def min_cover_path(centerline, a, step=None, iters=400, pin_ends=True, simplify_tol=1.0, disk_frac=0.85):
    """Compute an approximately minimum-length crack-covering path.

    Parameters
    ----------
    centerline : array-like, shape (M, 2), float
        Ordered ``(x, y)`` coordinates of the crack centerline.
    a : float
        Footprint radius in pixels.  The path must come within ``a`` of every
        centerline point.
    step : float or None, optional
        Resampling step for disk centres.  Defaults to ``a / 3``.
    iters : int, optional
        Maximum number of relaxation iterations.  The loop stops early when the
        maximum per-waypoint displacement falls below 0.05 pixels.
    pin_ends : bool, optional
        If ``True``, the first and last waypoints are fixed at the crack
        endpoints throughout the relaxation.
    simplify_tol : float, optional
        Perpendicular-distance tolerance (pixels) for the Ramer-Douglas-Peucker
        simplification applied after relaxation.
    disk_frac : float, optional
        Disk radius as a fraction of ``a`` (0 < disk_frac ≤ 1).  Values below
        1 provide a safety margin so that the simplified path still satisfies
        the coverage constraint after point removal.

    Returns
    -------
    numpy.ndarray, shape (K, 2), float
        Simplified ``(x, y)`` waypoints of the covering path.
    """
    a = float(a)
    r = disk_frac * a                            # disk radius < a: safety margin for inter-sample + simplify slack -> guarantees final max-gap <= a
    step = step if step else a / 3.0
    c = _resample(centerline, step)              # ordered disk centers
    if len(c) < 3:
        return c.copy()
    P = c.copy()                                 # init on the crack (feasible)
    lo = 1 if pin_ends else 0
    hi = len(P) - 1 if pin_ends else len(P)
    for _ in range(iters):
        maxmove = 0.0
        Pn = P.copy()
        for i in range(1, len(P) - 1):
            mid = 0.5 * (Pn[i - 1] + P[i + 1])   # Gauss-Seidel straighten
            v = mid - c[i]; d = np.hypot(*v)
            new = mid if d <= r else c[i] + v / d * r   # project into D(c_i, r)
            maxmove = max(maxmove, np.hypot(*(new - Pn[i])))
            Pn[i] = new
        P = Pn
        if maxmove < 0.05:
            break
    return _dp(P, simplify_tol)


def max_gap(centerline, path):
    """Return the maximum distance from any centerline point to the path.

    This is the coverage gap metric: if ``max_gap(centerline, path) <= a`` then
    the robot's footprint of radius ``a`` sweeps the entire crack.

    Parameters
    ----------
    centerline : array-like, shape (M, 2), float
        Ordered ``(x, y)`` crack centerline coordinates.
    path : array-like, shape (K, 2), float
        Path waypoints ``(x, y)``.

    Returns
    -------
    float
        Maximum Euclidean distance (pixels) from any resampled centerline
        point to the nearest point on the path polyline.
    """
    from shapely.geometry import LineString, Point
    ls = LineString(path)
    c = _resample(centerline, 3.0)
    return max(ls.distance(Point(p)) for p in c)


def length(path):
    """Return the arc length of a polyline.

    Parameters
    ----------
    path : array-like, shape (K, 2), float
        Ordered ``(x, y)`` waypoints.

    Returns
    -------
    float
        Sum of Euclidean segment lengths in pixels.
    """
    return float(np.hypot(*np.diff(np.asarray(path, float), axis=0).T).sum())


def min_cover_path_adaptive(centerline, a, lo=0.5, hi=1.0, bisect=6, **kw):
    """Compute the minimum-length crack-covering path with an adaptive disk fraction.

    Binary-searches for the largest ``disk_frac`` value in ``[lo, hi]`` whose
    resulting path still satisfies ``max_gap(path, centerline) <= a``.  Because
    ``max_gap`` is monotonically increasing in ``disk_frac``, bisection converges
    to the optimal fraction in ``bisect`` steps (resolution ``(hi - lo) / 2^bisect``).

    Parameters
    ----------
    centerline : array-like, shape (M, 2), float
        Ordered ``(x, y)`` crack centerline coordinates.
    a : float
        Footprint radius in pixels.
    lo : float, optional
        Lower bound on ``disk_frac`` (assumed to produce a feasible path).
    hi : float, optional
        Upper bound on ``disk_frac`` to search.
    bisect : int, optional
        Number of bisection steps.
    **kw
        Additional keyword arguments forwarded to ``min_cover_path``.

    Returns
    -------
    numpy.ndarray, shape (K, 2), float
        Covering path with the largest feasible disk fraction found.  Falls
        back to a simplified centerline if even ``lo`` is infeasible.
    """
    from shapely.geometry import LineString, Point
    cl = _resample(centerline, 3.0)
    def gap(P):
        ls = LineString(P)
        return max(ls.distance(Point(p)) for p in cl)
    best = min_cover_path(centerline, a, disk_frac=lo, **kw)   # lo assumed feasible
    if gap(best) > a:                                          # even lo violates -> fall back to centerline
        return _dp(centerline, 0.9 * a)
    L, H = lo, hi
    for _ in range(bisect):
        m = 0.5 * (L + H)
        P = min_cover_path(centerline, a, disk_frac=m, **kw)
        if gap(P) <= a:
            L, best = m, P
        else:
            H = m
    return best

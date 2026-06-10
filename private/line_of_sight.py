"""Line-of-sight (visibility) tests inside a boundary polygon.

Two functions are provided, suited to different callers:

:func:`line_of_sight`
    Parametric ray-vs-wall test, vectorized over multiple observer/target pairs.
    For each wall edge of the boundary polygon it solves for the beam parameter
    ``p`` (0 at the observer, 1 at the target) and the wall parameter ``q`` of
    the intersection.  A wall blocks the beam only when the crossing lies within
    the wall segment (``0 <= q <= 1``) at ``p >= 0``.  A target is considered
    visible when the nearest blocking crossing is at or beyond the target
    (``p >= 1``) **and** the segment midpoint falls inside the boundary.  Beams
    parallel to a wall are excluded from the wall check.  Used by
    :func:`~private.pathfinder.pathfinder` to build the visibility graph.

:func:`line_of_sight_poly`
    Polygon-based test.  A segment from observer to target is visible if and
    only if no part of it lies outside the boundary polygon (tested via
    :func:`~private.poly_utils.intersect_line`).  Used by the crack-planning
    module to build the crack visibility graph.
"""

import numpy as np

from private import poly_utils as pu


def line_of_sight(observer_state, current_target_node, external_boundaries):
    """Test visibility between observer/target pairs using a parametric ray-vs-wall check.

    For each (observer, target) pair, determines whether a straight-line beam
    from the observer to the target is unobstructed by the boundary walls and
    that the midpoint of the beam lies inside the boundary polygon.

    Parameters
    ----------
    observer_state : array-like, shape (N, 2) or (2,)
        Observer positions (x, y).
    current_target_node : array-like, shape (N, 2) or (2,)
        Target positions (x, y), same length as ``observer_state``.
    external_boundaries : array-like, shape (M, 2)
        Vertices of the boundary polygon (closed ring).

    Returns
    -------
    vis : numpy.ndarray, shape (N,)
        ``1.0`` where the target is visible from the observer, ``0.0`` otherwise.
    """
    obs = np.atleast_2d(np.asarray(observer_state, dtype=float))
    tgt = np.atleast_2d(np.asarray(current_target_node, dtype=float))
    B = np.asarray(external_boundaries, dtype=float)

    xo, yo = obs[:, 0], obs[:, 1]
    xt, yt = tgt[:, 0], tgt[:, 1]
    beam = tgt - obs
    N = obs.shape[0]
    M = B.shape[0]

    x1, y1 = B[:, 0], B[:, 1]
    x2 = np.concatenate([B[1:, 0], B[:1, 0]])
    y2 = np.concatenate([B[1:, 1], B[:1, 1]])

    bn = np.sqrt(beam[:, 0] ** 2 + beam[:, 1] ** 2)

    # Vectorized over both the N observer/target pairs AND the M wall edges.
    # Build (N, M) arrays; the three masked q assignments are applied in order
    # (the wy==0 / wx==0 masks overlap on a degenerate wx==wy==0 edge, where
    # the second assignment must win).
    wx = x2 - x1                                  # (M,)
    wy = y2 - y1
    wn = np.sqrt(wx ** 2 + wy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = (beam[:, 0][:, None] * wx[None, :] + beam[:, 1][:, None] * wy[None, :]) \
            / (bn[:, None] * wn[None, :])         # (N, M)
        indd = (ic != 1) & (ic != -1)
        num = wx[None, :] * (y1[None, :] - yo[:, None]) - wy[None, :] * (x1[None, :] - xo[:, None])
        den = wx[None, :] * (yt[:, None] - yo[:, None]) - wy[None, :] * (xt[:, None] - xo[:, None])
        p = num / den                             # (N, M)
        q = np.full((N, M), np.inf)
        pp = p >= 0
        wy0 = (wy == 0)[None, :]
        wx0 = (wx == 0)[None, :]
        m = pp & wy0 & indd
        q[m] = ((xo[:, None] - x1[None, :] + p * (xt[:, None] - xo[:, None])) / wx[None, :])[m]
        m = pp & wx0 & indd
        q[m] = ((yo[:, None] - y1[None, :] + p * (yt[:, None] - yo[:, None])) / wy[None, :])[m]
        m = pp & ~(wx0 | wy0) & indd
        q[m] = ((yo[:, None] - y1[None, :] + p * (yt[:, None] - yo[:, None])) / wy[None, :])[m]
        ok = (q >= 0) & (q <= 1)
        dist = np.where(ok, p, 0.0)               # (N, M)

    dist[dist == 0] = np.inf
    p_min = dist.min(axis=1)

    ind = p_min >= 1
    xm = np.zeros(N)
    ym = np.zeros(N)
    xm[ind] = 0.5 * (xo[ind] + xt[ind])
    ym[ind] = 0.5 * (yo[ind] + yt[ind])

    IN, ON = pu.inpolygon(xm, ym, B[:, 0], B[:, 1])
    vis = np.zeros(N)
    vis[IN | ON] = 1
    return vis



_EPS = 1e-9


def line_of_sight_poly(observer_node, target_node, external_boundaries):
    """Test visibility between observer/target pairs using a polygon containment check.

    A segment from observer to target is considered visible if and only if no
    part of it lies outside the boundary polygon.  Used by the crack-planning
    module to construct the crack visibility graph.

    Parameters
    ----------
    observer_node : array-like, shape (N, 2) or (2,)
        Observer positions (x, y).
    target_node : array-like, shape (N, 2) or (2,)
        Target positions (x, y), same length as ``observer_node``.
    external_boundaries : array-like, shape (2, M)
        Boundary polygon vertices as a ``[x; y]`` array (2 rows, M columns).

    Returns
    -------
    vis : numpy.ndarray, shape (N,)
        ``1.0`` where the target is visible from the observer, ``0.0`` otherwise.
    """
    on = np.atleast_2d(np.asarray(observer_node, dtype=float))
    tn = np.atleast_2d(np.asarray(target_node, dtype=float))
    eb = np.asarray(external_boundaries, dtype=float)
    ed = pu.polyshape(eb[0, :], eb[1, :])

    vis = np.ones(on.shape[0], dtype=float)
    for i in range(on.shape[0]):
        _, out = pu.intersect_line(ed, [on[i], tn[i]])
        if not out.is_empty and out.length > _EPS:
            vis[i] = 0
    return vis

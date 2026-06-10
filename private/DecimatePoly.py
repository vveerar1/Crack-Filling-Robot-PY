"""Polygon vertex decimation for the crack-filling coverage planner.

Provides :func:`DecimatePoly`, which simplifies a closed 2-D polygon contour by
iteratively removing the vertex whose removal causes the smallest accumulated
boundary-offset error.  Decimation stops when no remaining vertex's removal
error falls below a tolerance (mode 1) or when the contour has been reduced to
a target vertex count (mode 2).

This is the inverse operation to the midpoint-insertion in :mod:`refinePoly`:
refinement densifies a boundary for reliable critical-point detection; decimation
simplifies crack outlines and other dense contours for efficient downstream
graph construction.
"""

import numpy as np


def _poly_area(C):
    dx = C[:-1, 0] - C[1:, 0]
    dy = C[:-1, 1] + C[1:, 1]
    return abs(np.sum(dx * dy) / 2.0)


def _poly_perim(C):
    dE = np.sqrt(np.sum((C[1:] - C[:-1]) ** 2, axis=1))
    return float(np.sum(dE)), float(np.min(dE))


def _recompute_errors(V):
    """Compute the boundary-offset error introduced by removing the middle vertex V[1].

    Returns the squared distance from V[1] to its projection onto the replacement
    edge V[0]-V[2].  Uses scalar arithmetic for performance (this function is
    called in the innermost loop of :func:`DecimatePoly`).

    Parameters
    ----------
    V : array-like, shape (3, 2)
        Three consecutive polygon vertices: predecessor, candidate for removal, successor.

    Returns
    -------
    float
        Squared perpendicular distance (boundary-offset error).
    """
    v0x = V[0, 0]; v0y = V[0, 1]
    v1x = V[1, 0]; v1y = V[1, 1]
    v2x = V[2, 0]; v2y = V[2, 1]
    d31x = v2x - v0x; d31y = v2y - v0y
    dE = d31x * d31x + d31y * d31y
    if dE == 0.0:
        # V[0]==V[2] (coincident endpoints) -> the edge is a point at V[0], so the
        # projection p == V[0].  Force t=0 to avoid propagating 0/0 NaN into
        # downstream geometry (Python's min/max propagate NaN; we want finite t).
        dx = v0x - v1x; dy = v0y - v1y
        return float(dx * dx + dy * dy)
    d21x = v1x - v0x; d21y = v1y - v0y
    t = (d21x * d31x + d21y * d31y) / dE
    t = min(max(t, 0.0), 1.0)
    px = v0x + t * d31x; py = v0y + t * d31y
    dx = px - v1x; dy = py - v1y
    return float(dx * dx + dy * dy)


def DecimatePoly(C, opt=None, vis=False):
    """Simplify a closed 2-D polygon contour by iterative vertex removal.

    Repeatedly removes the vertex whose removal introduces the smallest
    accumulated boundary-offset error, either until no vertex falls below a
    tolerance (mode 1) or until the contour reaches a target vertex count
    (mode 2).

    Parameters
    ----------
    C : array-like, shape (N, 2)
        Closed contour vertices.  The last vertex may repeat the first
        (closing duplicate); it is handled automatically.
    opt : array-like [B_tol, mode], optional
        Decimation options:

        - ``mode = 1`` (default): stop when all remaining per-vertex errors
          exceed ``B_tol`` (boundary-offset tolerance in the same units as ``C``).
        - ``mode = 2``: stop when the contour has been reduced to
          ``round((N-1) * B_tol)`` vertices (``B_tol`` is a retain fraction in
          [0, 1]).

        If omitted, defaults to ``[Emin/2, 1]`` where ``Emin`` is the shortest
        edge length.
    vis : bool, optional
        Unused; kept for API compatibility.

    Returns
    -------
    C_out : numpy.ndarray, shape (M, 2)
        Decimated contour (closed; last vertex repeats the first).
    i_rem : numpy.ndarray of bool, shape (N,)
        ``True`` at the indices of the vertices that were removed from ``C``.
    CI : None
        Reserved; always ``None``.
    """
    C = np.array(C, dtype=float)
    N = C.shape[0]
    i_rem = np.zeros(N, dtype=bool)
    if N <= 4:
        return C, i_rem, None

    Po, Emin = _poly_perim(C)
    B_tol = Emin / 2.0
    Ao = _poly_area(C)
    No = N - 1

    if opt is not None and len(opt) > 0:
        B_tol = opt[0]
    if opt is None or len(opt) == 0:
        opt = [B_tol, 1]

    Nmin = 3
    if opt[1] == 2:
        Nmin = int(round((N - 1) * opt[0]))
        if (N - 1) == Nmin:
            return C, i_rem, None
        if Nmin < 3:
            Nmin = 3

    # remove repeating end-point
    C = C[:-1].copy()
    N = N - 1

    # initial distance offset errors (vectorized, circshift)
    Cprev = np.roll(C, 1, axis=0)
    Cnext = np.roll(C, -1, axis=0)
    D31 = Cnext - Cprev
    D21 = C - Cprev
    dE_new2 = np.sum(D31 ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.sum(D21 * D31, axis=1) / dE_new2
    # Cnext==Cprev -> dE_new2=0 -> 0/0=NaN here, but D31=0 so V=Cprev regardless.
    # Force t=0 to keep V finite (same as _recompute_errors).
    t = np.where(dE_new2 == 0.0, 0.0, t)
    t = np.clip(t, 0.0, 1.0)
    V = Cprev + t[:, None] * D31
    Err_D2 = np.sum((V - C) ** 2, axis=1)

    DEAA = np.zeros(N)
    idx_ret = list(range(1, N + 1))   # 1-based retained-vertex ids

    while True:
        idx_i = Err_D2 < B_tol
        if idx_i.sum() == 0 and N > Nmin and opt[1] == 2:
            B_tol = B_tol * np.sqrt(1.5)
            continue
        idxs = np.flatnonzero(idx_i)
        if idxs.size == 0 or N == Nmin:
            break
        N = N - 1

        i_min = int(np.argmin(Err_D2[idxs]))
        k0 = int(idxs[i_min])          # 0-based vertex to remove
        idx_i = k0 + 1                 # 1-based

        DEAA[k0] = DEAA[k0] + np.sqrt(Err_D2[k0])

        i1 = idx_i - 1
        if i1 < 1:
            i1 = N
        i3 = idx_i + 1
        if i3 > N:
            i3 = 1

        DEAA[i1 - 1] = DEAA[k0]
        DEAA[i3 - 1] = DEAA[k0]

        i1_1 = i1 - 1
        if i1_1 < 1:
            i1_1 = N
        i1_3 = i3
        i3_1 = i1
        i3_3 = i3 + 1
        if i3_3 > N:
            i3_3 = 1

        err_D1 = _recompute_errors(C[[i1_1 - 1, i1 - 1, i1_3 - 1]])
        err_D3 = _recompute_errors(C[[i3_1 - 1, i3 - 1, i3_3 - 1]])

        Err_D2[i1 - 1] = (np.sqrt(err_D1) + DEAA[i1 - 1]) ** 2
        Err_D2[i3 - 1] = (np.sqrt(err_D3) + DEAA[i3 - 1]) ** 2

        # Drop vertex k0 by slice-concat -- faster than np.delete for a
        # single removal inside a tight loop called tens of thousands of times.
        C = np.concatenate((C[:k0], C[k0 + 1:]))
        del idx_ret[k0]
        DEAA = np.concatenate((DEAA[:k0], DEAA[k0 + 1:]))
        Err_D2 = np.concatenate((Err_D2[:k0], Err_D2[k0 + 1:]))

    C = np.vstack([C, C[:1]])
    C_out = C

    i_rem[np.array(idx_ret, dtype=int) - 1] = True
    i_rem = ~i_rem
    i_rem[-1] = i_rem[0]
    return C_out, i_rem, None

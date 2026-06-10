"""Identify and classify endpoint pixels in a crack skeleton.

Given the accumulated crack skeleton seen so far (``BW3``) and a cleaned
reference copy (``BW``), this module locates all skeleton endpoints and
separates them into two classes:

- **Real endpoints** (``rP``) — pixels where the crack genuinely terminates.
  These are endpoint pixels that have a neighbour count other than 2 in the
  cleaned reference skeleton, indicating that they sit at a true tip rather
  than at a point where a branch was removed.

- **Continuing endpoints** (``cP``) — pixels where the crack continues beyond
  the currently sensed region.  These endpoint pixels have exactly 2 neighbours
  in the cleaned reference, meaning they lie on an ongoing chain that has been
  cut off at the sensor boundary.

The cleaned reference is derived from ``BW`` by iteratively removing branch
points (pixels with more than 2 neighbours) and the isolated pixels that result,
until no further changes occur.  This cleaning step ensures that only the
unbranched backbone of the crack is used when classifying endpoints.

All pixel coordinates are 0-based ``(row, col)``.
"""

import numpy as np

from private.utils import argwhere2d
from .bwmorph import neighbor_count_points, _neighbor_counts_at

_EMPTY = np.empty((0, 2), dtype=int)


def endP_ident(BW3, BW, org=None, bw3_fg=None):
    """Identify and classify crack skeleton endpoints.

    Parameters
    ----------
    BW3 : array-like, 2-D
        Accumulated crack skeleton (the union of all crack pixels detected so
        far).  Any non-zero value is foreground.
    BW : array-like, 2-D
        Reference crack skeleton used for endpoint classification.  Branch
        points and isolated pixels are removed from a copy of this mask
        before the neighbour count is evaluated.
    org : ignored
        Reserved parameter; accepted for interface compatibility.
    bw3_fg : numpy.ndarray, shape (N, 2), int, optional
        Precomputed foreground coordinates of ``BW3``.  When supplied, the
        function skips the full-image scan.  ``BW3`` must be 0/1 and
        ``bw3_fg`` must list exactly its non-zero pixels.

    Returns
    -------
    eP : numpy.ndarray, shape (E, 2), int
        All endpoint pixels of ``BW3`` — pixels with exactly one
        8-connected foreground neighbour.  Returns an empty array when
        ``BW3`` has five or fewer foreground pixels.
    rP : numpy.ndarray, shape (R, 2), int
        Real endpoints: the subset of ``eP`` whose neighbour count in the
        cleaned reference ``BW`` is not equal to 2.
    cP : numpy.ndarray, shape (C, 2), int
        Continuing endpoints: the subset of ``eP`` whose neighbour count in
        the cleaned reference ``BW`` equals 2.
    """
    # ``bw3_fg`` (optional): precomputed foreground coords of BW3. When the caller
    # already holds them, pass them to skip the full-image scan and output mask
    # allocation. BW3 must be 0/1 in that case.
    if bw3_fg is None:
        BW3 = np.asarray(BW3) > 0
        if BW3.sum() <= 5:
            return _EMPTY.copy(), _EMPTY.copy(), _EMPTY.copy()
        _, eP = neighbor_count_points(BW3, 1)             # 0-based (row, col), endpoints
    else:
        BW3 = np.asarray(BW3)                  # 0/1; used only for neighbour gather
        if bw3_fg.shape[0] <= 5:               # == BW3.sum() for 0/1 data
            return _EMPTY.copy(), _EMPTY.copy(), _EMPTY.copy()
        _, eP = neighbor_count_points(BW3, 1, pts_only=True, coords=bw3_fg)

    # Iteratively clean the reference: remove branch points, then the resulting
    # isolated single pixels, until no more changes occur. The foreground only ever
    # shrinks, so the live foreground coordinate set is maintained and neighbour
    # counts are evaluated only at those coordinates (sparse), keeping each pass O(nonzero).
    I = np.asarray(BW) > 0                           # fresh array (>0 copies) -> mutate in place
    coords = argwhere2d(I)                            # live foreground (row, col)
    while True:
        if coords.size:
            cnt = _neighbor_counts_at(I, coords)
            bpm = cnt > 2                             # branch points (count > 2)
            n_bp = int(bpm.sum())
            if n_bp:
                bp = coords[bpm]
                I[bp[:, 0], bp[:, 1]] = False
                coords = coords[~bpm]
        else:
            n_bp = 0
        if coords.size:
            cnt = _neighbor_counts_at(I, coords)     # on the post-branch-removal I
            spm = cnt == 0                            # single pixels (count == 0)
            n_sp = int(spm.sum())
            if n_sp:
                sp = coords[spm]
                I[sp[:, 0], sp[:, 1]] = False
                coords = coords[~spm]
        else:
            n_sp = 0
        if not (n_bp or n_sp):
            break

    if coords.size:                                  # coords IS the live foreground of I
        # evaluate neighbour counts only at the endpoint pixels (sparse) rather than
        # convolving the whole reference image.
        post = _neighbor_counts_at(I, eP) if eP.size else np.empty(0, dtype=int)
        rP = eP[post != 2] if eP.size else _EMPTY.copy()
        cP = eP[post == 2] if eP.size else _EMPTY.copy()
    else:
        rP = _EMPTY.copy()
        cP = _EMPTY.copy()

    return eP, rP, cP

"""Skeletonize a 2-D binary crack mask to a 1-pixel-wide medial axis.

This module implements Lee (1994) 3-D thinning to produce a topologically
correct 1-pixel-wide skeleton of a binary crack image.  The algorithm
iteratively removes simple points (pixels whose deletion preserves the
26-connectivity of the foreground) in six directional passes (North, South,
East, West, Up, Bottom) until no further removals are possible.  Euler
invariance is checked via an 8-octant look-up table before simplicity is
tested, so non-simple points and topology-changing deletions are never made.

The implementation processes all six border directions in natural order (1–6)
and scans column-major (column index advances in the outer loop).

Public API
----------
bwskel(image2d)
    Skeletonize a 2-D binary image.  Returns a bool array of the same shape.

compute_thin_image(vol, ...)
    Low-level configurable thinning kernel.  Accepts a 3-D padded volume and
    keyword arguments controlling border order and scan direction.  Useful for
    testing alternative configurations.
"""
import numpy as np

# Euler LUT (column dG26 of Table 2 of Lee et al. 1994); odd indices carry the values
_ARR = [1, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, -1,
        3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, 3, -1, 1, 1,
        3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1,
        -1, 1, 1, -1, 3, 1, 1, -1, -7, -1, -1, 1, -3, -1, -1, 1, -1,
        1, 1, -1, 3, 1, 1, -1, -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1,
        -1, 3, 1, 1, -1, -3, 3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3,
        1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1]
_EULER_LUT = np.zeros(256, dtype=np.intc)
_EULER_LUT[1::2] = _ARR

# octant index tables for the Euler-characteristic computation
_NEIGHB_IDX = [
    [2, 1, 11, 10, 5, 4, 14],
    [0, 9, 3, 12, 1, 10, 4],
    [8, 7, 17, 16, 5, 4, 14],
    [6, 15, 7, 16, 3, 12, 4],
    [20, 23, 19, 22, 11, 14, 10],
    [18, 21, 9, 12, 19, 22, 10],
    [26, 23, 17, 14, 25, 22, 16],
    [24, 25, 15, 16, 21, 22, 12],
]

# octree-labeling adjacency (octant -> (cube indices, adjacent octants per index))
_OCTREE = [
    ([0, 1, 3, 4, 9, 10, 12], [[], [2], [3], [2, 3, 4], [5], [2, 5, 6], [3, 5, 7]]),
    ([1, 4, 10, 2, 5, 11, 13], [[1], [1, 3, 4], [1, 5, 6], [], [4], [6], [4, 6, 8]]),
    ([3, 4, 12, 6, 7, 14, 15], [[1], [1, 2, 4], [1, 5, 7], [], [4], [7], [4, 7, 8]]),
    ([4, 5, 13, 7, 15, 8, 16], [[1, 2, 3], [2], [2, 6, 8], [3], [3, 7, 8], [], [8]]),
    ([9, 10, 12, 17, 18, 20, 21], [[1], [1, 2, 6], [1, 3, 7], [], [6], [7], [6, 7, 8]]),
    ([10, 11, 13, 18, 21, 19, 22], [[1, 2, 5], [2], [2, 4, 8], [5], [5, 7, 8], [], [8]]),
    ([12, 14, 15, 20, 21, 23, 24], [[1, 3, 5], [3], [3, 4, 8], [5], [5, 6, 8], [], [8]]),
    ([13, 15, 16, 21, 22, 24, 25], [[2, 4, 6], [3, 4, 7], [4], [5, 6, 7], [6], [7], []]),
]

# 27-neighborhood offsets in the canonical order of skimage's get_neighborhood()
_OFF = [
    (-1, -1, -1), (-1, 0, -1), (-1, 1, -1),
    (-1, -1, 0), (-1, 0, 0), (-1, 1, 0),
    (-1, -1, 1), (-1, 0, 1), (-1, 1, 1),
    (0, -1, -1), (0, 0, -1), (0, 1, -1),
    (0, -1, 0), (0, 0, 0), (0, 1, 0),
    (0, -1, 1), (0, 0, 1), (0, 1, 1),
    (1, -1, -1), (1, 0, -1), (1, 1, -1),
    (1, -1, 0), (1, 0, 0), (1, 1, 0),
    (1, -1, 1), (1, 0, 1), (1, 1, 1),
]


def _start_octant(i):
    if i in (0, 1, 3, 4, 9, 10, 12): return 1
    if i in (2, 5, 11, 13): return 2
    if i in (6, 7, 14, 15): return 3
    if i in (8, 16): return 4
    if i in (17, 18, 20, 21): return 5
    if i in (19, 22): return 6
    if i in (23, 24): return 7
    return 8
_START_OCT = [_start_octant(i) for i in range(26)]


def _octree_labeling(octant, label, cube):
    """Label all foreground octants connected to *octant* with *label* (iterative)."""
    stack = [octant]
    while stack:
        oc = stack.pop()
        indices, lists = _OCTREE[oc - 1]
        for idx, new_octs in zip(indices, lists):
            if cube[idx] == 1:
                cube[idx] = label
                stack.extend(new_octs)


def _is_simple_point(nb):
    """Return True if the centre voxel is a simple point (safe to delete).

    A voxel is simple when removing it leaves the 26-connected foreground
    component of its 3x3x3 neighbourhood unchanged — i.e. the 26-neighbourhood
    (centre excluded) has exactly one 26-connected foreground component.

    Parameters
    ----------
    nb : array-like, length 27
        Flattened 3x3x3 neighbourhood values (centre at index 13).
    """
    cube = np.empty(26, dtype=np.intc)
    cube[:13] = nb[:13]
    cube[13:] = nb[14:]
    label = 2
    for i in range(26):
        if cube[i] == 1:
            _octree_labeling(_START_OCT[i], label, cube)
            label += 1
            if label - 2 >= 2:
                return False
    return True


def _euler_invariant_vec(nbs):
    """Return True for each voxel whose deletion preserves the Euler number.

    Parameters
    ----------
    nbs : numpy.ndarray, shape (K, 27)
        Flattened 3x3x3 neighbourhoods for K candidate voxels.

    Returns
    -------
    numpy.ndarray, bool, shape (K,)
        ``True`` where deletion is Euler-invariant (Euler characteristic change = 0).
    """
    euler = np.zeros(nbs.shape[0], dtype=np.intc)
    for octant in range(8):
        idxs = _NEIGHB_IDX[octant]
        n = np.ones(nbs.shape[0], dtype=np.intc)
        for j in range(7):
            n |= (nbs[:, idxs[j]] == 1) * (1 << (7 - j))
        euler += _EULER_LUT[n]
    return euler == 0


# direction -> (axis, sign): which neighbor must be background for a border point
#   1=N: c-1 ; 2=S: c+1 ; 3=E: r+1 ; 4=W: r-1 ; 5=U: p+1 ; 6=B: p-1
_DIR = {1: (2, -1), 2: (2, +1), 3: (1, +1), 4: (1, -1), 5: (0, +1), 6: (0, -1)}

# C-order-flattened 3x3x3 block (dp slow, dr, dc fast) -> _OFF order
_PERM27 = np.array([(dp + 1) * 9 + (dr + 1) * 3 + (dc + 1) for (dp, dr, dc) in _OFF])


def _gather_neighborhood(img, P, R, C):
    nbs = np.empty((P.shape[0], 27), dtype=img.dtype)
    for k, (dp, dr, dc) in enumerate(_OFF):
        nbs[:, k] = img[P + dp, R + dr, C + dc]
    return nbs


def _nb_single(img, p, r, c):
    return img[p - 1:p + 2, r - 1:r + 2, c - 1:c + 2].reshape(27)[_PERM27]


def compute_thin_image(img, borders=(4, 3, 2, 1, 5, 6), scan_order=(0, 1, 2),
                       scan_rev=(False, False, False), force_num_borders=None):
    """Iteratively thin a 3-D binary volume using Lee (1994) directional thinning.

    Each iteration cycles through up to six directional passes (border
    directions 1=North, 2=South, 3=East, 4=West, 5=Up, 6=Bottom).  In each
    pass, candidate border points are collected, filtered by the Euler
    invariance criterion, and then tested one by one for simplicity.  Simple
    points are deleted.  The loop stops when a full cycle produces no
    deletions in any direction.

    Parameters
    ----------
    img : numpy.ndarray, uint8, shape (D, H, W)
        Zero-padded 3-D volume with foreground voxels set to 1.  The function
        modifies this array in place and also returns it.
    borders : tuple of int, optional
        Sequence of border direction codes, controlling which directions are
        processed and in what order.  Default reproduces scikit-image ordering.
    scan_order : tuple of int (length 3), optional
        Axis indices (0=depth, 1=row, 2=col) giving the sort order when
        collecting candidate points within a border pass.  Index 0 is the
        outermost (primary) sort axis.
    scan_rev : tuple of bool (length 3), optional
        Whether to reverse each sort axis.
    force_num_borders : int or None, optional
        Override the number of active border directions.  When ``None``,
        defaults to 4 for a single-slice volume (2-D input) or 6 otherwise.

    Returns
    -------
    numpy.ndarray, uint8, shape (D, H, W)
        The thinned volume (same object as ``img``, modified in place).
    """
    img = np.ascontiguousarray(img.astype(np.uint8))
    shp = img.shape
    num_borders = (4 if shp[0] == 3 else 6) if force_num_borders is None else force_num_borders
    borders = list(borders)
    axes, revs = list(scan_order), list(scan_rev)

    def collect(curr_border):
        ax, sign = _DIR[curr_border]
        fg = img == 1
        idx_dst = [slice(None)] * 3
        idx_src = [slice(None)] * 3
        if sign == +1:
            idx_dst[ax] = slice(0, shp[ax] - 1)
            idx_src[ax] = slice(1, shp[ax])
        else:
            idx_dst[ax] = slice(1, shp[ax])
            idx_src[ax] = slice(0, shp[ax] - 1)
        neighbor_bg = np.zeros_like(fg)
        neighbor_bg[tuple(idx_dst)] = (img[tuple(idx_src)] == 0)
        cand = fg & neighbor_bg
        cand[0, :, :] = cand[-1, :, :] = False
        cand[:, 0, :] = cand[:, -1, :] = False
        cand[:, :, 0] = cand[:, :, -1] = False

        P, R, C = np.nonzero(cand)
        if P.size == 0:
            return P, R, C
        coords = [P, R, C]
        keys = [(-coords[a].astype(np.int64)) if rev else coords[a].astype(np.int64)
                for a, rev in zip(axes, revs)]
        order = np.lexsort(keys[::-1])      # axes[0] = outermost / primary
        P, R, C = P[order], R[order], C[order]

        nbs = _gather_neighborhood(img, P, R, C)
        keep = (nbs.sum(axis=1) != 2) & _euler_invariant_vec(nbs)   # not endpoint & Euler
        P, R, C, nbs = P[keep], R[keep], C[keep], nbs[keep]
        simple = np.array([_is_simple_point(nbs[i]) for i in range(nbs.shape[0])], bool)
        return P[simple], R[simple], C[simple]

    unchanged = 0
    while unchanged < num_borders:
        unchanged = 0
        for j in range(num_borders):
            P, R, C = collect(borders[j])
            no_change = True
            Pl, Rl, Cl = P.tolist(), R.tolist(), C.tolist()
            for i in range(len(Pl)):
                p, r, c = Pl[i], Rl[i], Cl[i]
                if img[p, r, c] != 1:
                    continue
                if _is_simple_point(_nb_single(img, p, r, c)):
                    img[p, r, c] = 0
                    no_change = False
            if no_change:
                unchanged += 1
    return img


# All 6 borders, natural order, column-major scan (column index is outermost sort axis).
_SKEL_KW = dict(borders=(1, 2, 3, 4, 5, 6), force_num_borders=6,
                scan_order=(0, 2, 1), scan_rev=(False, False, False))


def bwskel(image2d):
    """Skeletonize a 2-D binary crack mask to a 1-pixel-wide medial axis.

    The image is embedded in a single-slice 3-D volume, thinned with Lee (1994)
    directional thinning using all six border directions in natural order and a
    column-major candidate scan, then the result is extracted back to 2-D.

    Parameters
    ----------
    image2d : array-like, 2-D
        Binary crack mask; any non-zero value is treated as foreground.

    Returns
    -------
    numpy.ndarray, bool, shape (H, W)
        1-pixel-wide skeleton of the input mask.
    """
    img = (np.asarray(image2d) != 0).astype(np.uint8)
    vol = np.pad(img[np.newaxis, ...], 1, mode="constant")
    out = compute_thin_image(vol, **_SKEL_KW)
    return out[1:-1, 1:-1, 1:-1][0].astype(bool)

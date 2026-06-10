"""Binary-image morphology operations for crack skeleton analysis.

This module provides four groups of operations applied to 1-pixel-wide crack
skeletons:

**Thinning and spur removal**

- ``bwmorph_thin`` — Guo-Hall (1989) two-subiteration parallel thinning, run for
  up to *n* iterations (or to convergence). Thins a binary mask without breaking
  connectivity. Note: this is a thinning pass, not a full skeletonization.
- ``bwmorph_spur`` — Remove diagonal spur tips: a foreground pixel is erased if
  it has exactly one 8-connected foreground neighbour and that neighbour is
  diagonal. Applied *n* times, so only the shortest isolated stubs are removed.

**Neighbour-count feature detection**

- ``neighbor_count_points`` — Classify foreground pixels by their 8-neighbour
  count: 0 neighbours → isolated pixel, 1 neighbour → endpoint, more than 2
  neighbours → branch/intersection point. Returns both a labelled mask and a
  coordinate list of the matching pixels.

**Look-up table endpoint and branchpoint detection**

- ``endpoints`` — Detect skeleton endpoints using a 512-entry 3x3 look-up table
  (LUT). Each pixel's 3x3 neighbourhood is encoded as a 9-bit index; the LUT
  returns whether that pattern is an endpoint.
- ``branchpoints`` — Detect skeleton branch points using a set of 512-entry 3x3
  LUTs. Pixels where three or more branches meet are identified; the algorithm
  further filters out endpoints and resolves ambiguous multi-branch junctions so
  that each physical junction contributes a single branch point.
- ``bwlookup`` — Apply an arbitrary 512-entry 3x3 LUT to a binary image
  (the primitive used by ``endpoints`` and ``branchpoints``).

The LUTs are loaded from ``bwmorph_luts.npz`` (bundled alongside this module).
"""
import os

import numpy as np
from scipy import ndimage as ndi
from private.utils import argwhere2d


def bwmorph_thin(image, n_iter=None):
    """Thin a binary image using Guo-Hall (1989) parallel thinning.

    Iteratively removes foreground pixels that can be deleted without breaking
    8-connectivity, thinning the mask toward a 1-pixel-wide skeleton. Each
    iteration consists of two sub-iterations (Guo-Hall's odd/even passes).

    Parameters
    ----------
    image : array-like, 2-D
        Binary input image (any non-zero value is foreground).
    n_iter : int or None, optional
        Maximum number of thinning iterations. ``None`` runs to convergence.

    Returns
    -------
    numpy.ndarray, bool, shape (H, W)
        Thinned binary mask; ``True`` where foreground pixels remain.
    """
    from skimage.morphology import thin as _thin
    out = _thin(np.asarray(image) > 0, max_num_iter=n_iter)
    return out.astype(bool)


# Spur detection: 3x3 neighborhood encoded as a power-of-two weighted convolution, then LUT.
_SPUR_W = np.array([[1, 8, 64], [2, 16, 128], [4, 32, 256]], dtype=np.uint16)


def _spur_lut():
    lut = np.zeros(512, dtype=bool)
    lut[16:32] = True                 # identity over center bit, base block
    # build full identity: output = center bit (weight 16) set
    idx = np.arange(512)
    lut = (idx & 16) > 0
    lut[[17, 20, 80, 272]] = False    # diagonal-spur-tip exceptions (1-indexed: 18,21,81,273)
    return lut


_SPUR_LUT = _spur_lut()


def bwmorph_spur(BW, n):
    """Remove diagonal spur tips from a binary skeleton, up to *n* times.

    A foreground pixel is a diagonal spur tip when it has exactly one
    8-connected foreground neighbour and that neighbour is diagonally adjacent.
    Only the four rotation/reflection variants of this pattern are matched; all
    other pixels are left unchanged.  Applying the operation *n* times removes
    stubs of up to *n* pixels, leaving longer branches intact.

    Parameters
    ----------
    BW : array-like, 2-D
        Binary skeleton image (any non-zero value is foreground).
    n : int
        Number of spur-removal passes.

    Returns
    -------
    numpy.ndarray, int (0/1), shape (H, W)
        Skeleton with diagonal spur tips removed.
    """
    BW = (np.asarray(BW) > 0).astype(np.uint16)
    for _ in range(int(n)):
        N = ndi.correlate(BW, _SPUR_W, mode='constant')
        keep = np.take(_SPUR_LUT, N)
        new = (BW > 0) & keep
        if np.array_equal(new, BW > 0):
            break
        BW = new.astype(np.uint16)
    return (BW > 0).astype(int)


_NEIGHBORS8 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=int)

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)]


def _neighbor_counts_at(BW, coords):
    """Count 8-connected foreground neighbours at a sparse set of pixel locations.

    Parameters
    ----------
    BW : numpy.ndarray, shape (H, W)
        Binary image (0/1 or bool). Out-of-bounds neighbours are treated as 0.
    coords : numpy.ndarray, shape (N, 2), int
        Pixel locations ``(row, col)`` at which to compute counts.

    Returns
    -------
    numpy.ndarray, shape (N,), int
        Number of foreground 8-neighbours at each location in ``coords``.
    """
    H, W = BW.shape
    r, c = coords[:, 0], coords[:, 1]
    cnt = np.zeros(coords.shape[0], dtype=int)
    for dr, dc in _OFFSETS:
        rr = r + dr
        cc = c + dc
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        idx = np.flatnonzero(ok)
        if idx.size:
            cnt[idx] += BW[rr[idx], cc[idx]]      # bool adds as 0/1 into the int accumulator
    return cnt


def neighbor_count_points(BW, operation, pts_only=False, coords=None):
    """Locate foreground pixels whose 8-neighbour count matches *operation*.

    Parameters
    ----------
    BW : array-like, 2-D
        Binary image.  Any non-zero value is foreground.
    operation : int
        Target neighbour count: ``0`` selects isolated pixels (0 neighbours),
        ``1`` selects endpoints (1 neighbour), ``2`` selects branch/intersection
        points (strictly more than 2 neighbours).
    pts_only : bool, optional
        If ``True``, skip building the full-image output mask and return ``None``
        in its place.  Use this when only the coordinate list is needed.
    coords : numpy.ndarray, shape (N, 2), int, optional
        Precomputed foreground coordinates of ``BW``.  When supplied, the
        function skips the full-image scan and evaluates neighbour counts only
        at these locations.  ``BW`` must be 0/1 and ``coords`` must list exactly
        its non-zero pixels.

    Returns
    -------
    BW_out : numpy.ndarray, int (0/1), shape (H, W), or None
        Mask set to 1 at the matching pixels.  ``None`` when ``pts_only=True``.
    pts : numpy.ndarray, shape (M, 2), int
        Coordinates ``(row, col)`` of the matching pixels, ordered by column
        then row (consistent with column-major enumeration).
    """
    if coords is None:
        BW = np.asarray(BW) > 0
        # Only set pixels can be selected and the 8-neighbour count depends only on
        # neighbours within 1px -- so evaluate counts at the nonzero pixels directly
        # (sparse gather), avoiding a full-image convolution. Identical result,
        # O(nonzero) instead of O(image) for the sparse inner-loop skeletons.
        coords = argwhere2d(BW)
    else:
        BW = np.asarray(BW)              # caller-guaranteed 0/1; no full bool-conv
        coords = np.asarray(coords, dtype=int)
    if coords.size == 0:
        out = None if pts_only else np.zeros(BW.shape, dtype=int)
        return out, np.empty((0, 2), dtype=int)
    counts = _neighbor_counts_at(BW, coords)

    keep = (counts == operation) if operation < 2 else (counts > operation)
    pts = coords[keep]
    if pts_only:
        BW_out = None
    else:
        BW_out = np.zeros(BW.shape, dtype=int)
        if pts.size:
            BW_out[pts[:, 0], pts[:, 1]] = 1
    if pts.size:
        # order by column then row (primary col, secondary row)
        pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    else:
        pts = np.empty((0, 2), dtype=int)
    return BW_out, pts


_LUTS = np.load(os.path.join(os.path.dirname(__file__), "bwmorph_luts.npz"))
_LUT_ENDPOINTS = _LUTS["endpoints"].astype(bool)
_LUT_BRANCH = _LUTS["branchpoints"].astype(bool)
_LUT_DILATE = _LUTS["dilate"].astype(bool)
_LUT_BACKCOUNT4 = _LUTS["backcount4"].astype(int)


def _lut_index(bw):
    """Compute the 3x3 LUT index at every pixel.

    Each pixel's 3x3 neighbourhood is encoded as a 9-bit integer: the bit
    weight of position (r, c) within the 3x3 block is ``2^((r-1) + 3*(c-1))``,
    with the top-left corner as the least significant bit.

    Parameters
    ----------
    bw : array-like, 2-D
        Binary image.

    Returns
    -------
    numpy.ndarray, int64, shape (H, W)
        LUT index (0–511) for each pixel.
    """
    bw = (np.asarray(bw) > 0).astype(np.int64)
    H, W = bw.shape
    P = np.pad(bw, 1)
    idx = np.zeros((H, W), np.int64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            idx += P[1 + dr:1 + dr + H, 1 + dc:1 + dc + W] << ((dr + 1) + 3 * (dc + 1))
    return idx


def bwlookup(bw, lut):
    """Apply a 512-entry 3x3 look-up table to a binary image.

    Each pixel's 3x3 neighbourhood is encoded as a 9-bit index (see
    ``_lut_index``) and the corresponding LUT entry is written to the output.

    Parameters
    ----------
    bw : array-like, 2-D
        Binary input image.
    lut : array-like, shape (512,)
        Look-up table; element *i* is the output value for neighbourhood code
        *i*.

    Returns
    -------
    numpy.ndarray, shape (H, W)
        Per-pixel LUT output, same dtype as ``lut``.
    """
    return np.asarray(lut)[_lut_index(bw)]


def endpoints(bw):
    """Detect skeleton endpoints using a 3x3 look-up table.

    An endpoint is a foreground pixel with exactly one 8-connected foreground
    neighbour.  The result is equivalent to thresholding the 8-neighbour count
    at 1, but the LUT implementation also handles border cases consistently.

    Parameters
    ----------
    bw : array-like, 2-D
        Binary skeleton image.

    Returns
    -------
    numpy.ndarray, bool, shape (H, W)
        ``True`` at every endpoint pixel.
    """
    return _LUT_ENDPOINTS[_lut_index(bw)]


def branchpoints(bw):
    """Detect skeleton branch points using a set of 3x3 look-up tables.

    A branch point is a foreground pixel where three or more skeleton branches
    meet.  The detection uses four LUTs: one to find candidate branch pixels,
    one to count background 4-connected components in the 3x3 neighbourhood,
    and two more to resolve ambiguous multi-branch junctions so that each
    physical junction contributes exactly one branch-point pixel.

    Parameters
    ----------
    bw : array-like, 2-D
        Binary skeleton image.

    Returns
    -------
    numpy.ndarray, bool, shape (H, W)
        ``True`` at every branch point pixel.
    """
    idx = _lut_index(bw)
    C = _LUT_BRANCH[idx]
    B = _LUT_BACKCOUNT4[idx]
    E = (B == 1)
    FC = (~E) & C
    Vp = (B == 2) & (~E)
    Vq = (B > 2) & (~E)
    D = _LUT_DILATE[_lut_index(Vq)]
    M = (FC & Vp) & D
    return FC & (~M)

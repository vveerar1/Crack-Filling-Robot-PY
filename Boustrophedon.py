"""Single-region boustrophedon coverage orchestrator.

Boustrophedon drives the per-cell zig-zag sweep over a single decomposed region.
It walks the Reeb graph traversal order (``Path``); for each cell edge it calls
:func:`BoustrophedonPath` to generate that cell's back-and-forth sweep, then
links consecutive cell paths with intermediate waypoints at the sensor spacing.
The result is ``PathEdge``, the ordered (x, y) coverage path for the region.

This module handles the single-region case.  For multi-cell regions with
cell-connection optimisation, see :mod:`Boustrophedon_CellCon`.
"""

import math

import numpy as np

from private import poly_utils as pu
from private.utils import addPtsLin
from BoustrophedonPath import BoustrophedonPath

_XMAX, _YMAX = 3048, 2898


def _is_reeb_edge(EE, reebEdge):
    """Return the index of edge ``EE`` (either orientation) in ``reebEdge``, or ``None``."""
    for k, e in enumerate(reebEdge):
        if (e[0] == EE[0] and e[1] == EE[1]) or (e[0] == EE[1] and e[1] == EE[0]):
            return k
    return None


def _clamp(subXY, sensor):
    subXY = np.array(subXY, dtype=float)
    subXY[subXY[:, 0] < sensor, 0] = sensor
    subXY[subXY[:, 0] > _XMAX - sensor, 0] = _XMAX - sensor
    subXY[subXY[:, 1] < sensor, 1] = sensor
    subXY[subXY[:, 1] > _YMAX - sensor, 1] = _YMAX - sensor
    return subXY


def Boustrophedon(Path, splitReg, see, seP, init, wall_fol, known, sim,
                  reebEdge, reebCell, allNode, s, a):
    """Generate the boustrophedon coverage path for a single decomposed region.

    Iterates over the Reeb traversal order ``Path``.  For each Reeb-cell edge the
    corresponding cell is swept with a back-and-forth (boustrophedon) pattern at
    sensor spacing ``s / sqrt(2)``.  Non-cell (connection) edges simply link the
    previous endpoint to the next node.  Consecutive sub-paths are joined with
    intermediate waypoints at spacing ``s``.

    Parameters
    ----------
    Path : array-like, shape (N, 2)
        Reeb traversal order — each row is a pair of critical-point node indices
        ``[start, end]`` defining one cell or connection edge.
    splitReg : list
        Coverage cells produced by the Morse Cell Decomposition (MCD), indexed by
        the entries in ``reebCell``.
    see : ignored
        Legacy input; not used (direction is chosen per-cell automatically).
    seP : ignored
        Legacy input; not used.
    init : array-like, shape (2,)
        Initial robot position (x, y).  Pass ``[0, 0]`` if no prior position.
    wall_fol : array-like, shape (N,)
        Per-edge flag indicating whether wall-following is required (1) or a
        standard zig-zag sweep is used (0).
    known : bool
        ``True`` for the offline (SCC) planner; ``False`` for the online
        (OnlineSCC) planner.  Affects cell-path stitching behaviour.
    sim : bool
        When ``True``, disables boundary clamping (used in simulation mode).
    reebEdge : array-like, shape (E, 2)
        Reeb graph edges as pairs of critical-point node indices.
    reebCell : array-like, shape (E,)
        Index into ``splitReg`` for each Reeb edge.
    allNode : array-like, shape (M, 2)
        Critical-point coordinates in (row, col) order as produced by MCD.
    s : float
        Sensor radius in pixels.  The zig-zag sweep spacing is ``s / sqrt(2)``.
    a : float
        Robot footprint radius in pixels.

    Returns
    -------
    PathEdge : ndarray, shape (K, 2)
        Ordered (x, y) coverage waypoints for the entire region.
    """
    reebEdge = np.atleast_2d(np.asarray(reebEdge, dtype=int)).copy()
    reebCell = np.asarray(reebCell, dtype=int)
    allNode = np.atleast_2d(np.asarray(allNode, dtype=float))
    Path = np.atleast_2d(np.asarray(Path, dtype=int))
    sensor = s / math.sqrt(2)
    fl_see = False

    PathEdge = np.atleast_2d(np.asarray(init, dtype=float)) if np.ndim(init) and np.size(init) > 1 else np.empty((0, 2))
    o = -1

    for i in range(Path.shape[0]):
        EE = Path[i, :2]
        ind = _is_reeb_edge(EE, reebEdge)
        if ind is not None:
            o += 1
            cell = splitReg[int(reebCell[ind])]
            reebEdge[ind] = [-99, -99]
            subXY, flag = BoustrophedonPath(cell, cell, EE, sensor, [], init,
                                            int(wall_fol[i]), known, fl_see, allNode, s, a)
            if not sim:
                subXY = _clamp(subXY, sensor)
        else:
            if PathEdge.shape[0]:
                subXY = PathEdge[-1:].copy()
            else:
                subXY = np.empty((0, 2))

        if subXY is not None and len(subXY):
            subXY = np.atleast_2d(subXY)
            if PathEdge.shape[0]:
                mx, my = addPtsLin([PathEdge[-1, 0], subXY[0, 0]],
                                   [PathEdge[-1, 1], subXY[0, 1]], s)
                link = np.column_stack([mx, my]) if len(mx) else np.empty((0, 2))
                PathEdge = np.vstack([PathEdge, link, subXY])
            else:
                PathEdge = np.vstack([PathEdge, subXY])
            init = subXY[-1]

    return PathEdge

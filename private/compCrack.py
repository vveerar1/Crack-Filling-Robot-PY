"""Trace a crack skeleton into ordered polylines (crack segments).

Given a 1-pixel-wide crack skeleton image and a list of endpoint pixels,
this module walks the skeleton from each endpoint and records the traversed
pixel coordinates as an ordered polyline.  The result is a set of crack
segments, each represented as a sequence of (row, col) coordinates from one
endpoint to the other, together with compact summary arrays used by downstream
planners.

Two tracer implementations are provided and dispatched automatically:

``compCrack`` (online-planner tracer)
    The main entry point for the crack-filling planner.  Handles skeletons
    that consist entirely of simple chains (no pixel with more than two
    8-connected foreground neighbours).  Each connected component is walked
    from its first endpoint to the other using a sparse adjacency structure
    built from the foreground coordinates alone, without reading the full
    image on every step.  Falls back to the sequential tracer if a genuine
    junction pixel is encountered.

``compCrack_branching`` (offline-planner tracer)
    Used when the skeleton may contain branch points (pixels where three or
    more branches meet).  At each genuine junction the current segment is
    ended at the branch pixel and each unvisited outgoing branch is pushed
    onto the work queue as a new segment start.  This decomposes a branchy
    skeleton into a set of endpoint-to-branchpoint and
    branchpoint-to-branchpoint segments, which is the representation required
    by the offline crack-graph builder.

Both tracers return the same four outputs:

- ``crackRaw`` : list of ``(M, 2)`` int arrays, one per segment, each row a
  ``(row, col)`` pixel coordinate.  The final row repeats the last pixel
  (endpoint duplicate).
- ``line`` : ``(K, 4)`` int array; row *k* is
  ``[start_row, start_col, end_row, end_col]`` for segment *k*.
- ``pointX`` : ``(K, 24)`` float array; each segment uniformly down-sampled
  to 24 row-coordinates with endpoints forced.
- ``pointY`` : ``(K, 24)`` float array; corresponding column-coordinates.

All coordinates are 0-based.
"""

import numpy as np

from private.utils import mround, spdist, argwhere2d

# 8-connectivity direction offsets (drow, dcol), standard order.
_OFFS = np.array([[-1, -1], [-1, 0], [-1, 1],
                  [0, -1], [0, 1],
                  [1, -1], [1, 0], [1, 1]])


def _trace_fg(fg, endP, H, W, dir_map):
    """Walk simple-chain skeleton components using sparse foreground coordinates.

    Builds 8-neighbour adjacency among the ``N`` foreground pixels using a
    sorted linear-index array and binary search, then walks each connected
    chain from its first endpoint to the other.  Never reads the full image
    after the initial foreground coordinate array is supplied.

    Parameters
    ----------
    fg : numpy.ndarray, shape (N, 2), int
        Foreground pixel coordinates ``(row, col)``.
    endP : numpy.ndarray, shape (E, 2), int
        Skeleton endpoint coordinates used to determine walk start points and
        directions.
    H, W : int
        Image height and width (for bounds checking).
    dir_map : numpy.ndarray, shape (8, 2), int
        8-connectivity direction offsets ``(drow, dcol)``.

    Returns
    -------
    list of numpy.ndarray, shape (M, 2)
        Ordered pixel coordinates for each chain, or ``None`` if any pixel has
        more than two 8-connected neighbours (signals the caller to fall back
        to the sequential tracer).
    """
    N = fg.shape[0]
    if N == 0:
        return []
    lin = fg[:, 0].astype(np.int64) * W + fg[:, 1]      # column-major-safe linear ids
    order = np.argsort(lin, kind="stable")
    lin_s = lin[order]
    r = fg[:, 0]
    c = fg[:, 1]
    # nbr[i,k] = node index of pixel i's k-th 8-neighbour (or -1 if none/OOB)
    nbr = -np.ones((N, 8), dtype=np.int64)
    for k in range(8):
        rr = r + dir_map[k, 0]
        cc = c + dir_map[k, 1]
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        nl = rr.astype(np.int64) * W + cc
        pos = np.clip(np.searchsorted(lin_s, nl), 0, N - 1)
        match = ok & (lin_s[pos] == nl)
        nbr[match, k] = order[pos[match]]
    if (nbr >= 0).sum(axis=1).max() > 2:                # a true junction -> fall back
        return None
    # node id of each endpoint (in endP order), or -1 if not a foreground pixel
    eln = endP[:, 0].astype(np.int64) * W + endP[:, 1]
    epos = np.clip(np.searchsorted(lin_s, eln), 0, N - 1)
    inb = ((endP[:, 0] >= 0) & (endP[:, 0] < H) & (endP[:, 1] >= 0) & (endP[:, 1] < W)
           & (lin_s[epos] == eln))
    epnode = np.where(inb, order[epos], -1)
    nbrl = nbr.tolist()                                 # python lists -> fast scalar walk
    done = np.zeros(N, dtype=bool)
    out = []
    # emit one segment per component, started from the first of its endpoints to appear in endP.
    for sid in epnode.tolist():
        if sid < 0 or done[sid]:
            continue
        nodes = [sid]
        done[sid] = True
        prev = -1
        cur = sid
        while True:                                     # one unvisited neighbour per step
            row = nbrl[cur]
            nxt = -1
            for k in range(8):                          # first in dir order == b.index(1)
                nid = row[k]
                if nid >= 0 and nid != prev:
                    nxt = nid
                    break
            if nxt < 0:
                break
            nodes.append(nxt)
            done[nxt] = True
            prev = cur
            cur = nxt
        if len(nodes) < 2:                              # isolated pixel -> no segment
            continue
        ch = fg[np.asarray(nodes)]
        out.append(np.vstack([ch, ch[-1:]]))            # [start; path; EndPoint(dup)]
    return out


def _nbr_vec(I, row, col):
    """Return the 8-neighbour presence vector at ``(row, col)``.

    Returns a length-8 list of 0/1 values in the standard 8-connectivity
    direction order.  Out-of-bounds positions are treated as background (0).
    """
    h, w = I.shape
    if 0 < row < h - 1 and 0 < col < w - 1:
        f = I[row - 1:row + 2, col - 1:col + 2].ravel().tolist()
        return [f[0], f[1], f[2], f[3], f[5], f[6], f[7], f[8]]
    b = [0] * 8
    for k in range(8):
        r = row + _OFFS[k, 0]
        c = col + _OFFS[k, 1]
        if 0 <= r < h and 0 <= c < w and I[r, c]:
            b[k] = 1
    return b


def _dsearchn(endP, pt):
    """Return the index and distance of the nearest row in ``endP`` to point ``pt``."""
    d = spdist(pt, endP)
    f = int(np.argmin(d))
    return f, float(d[f])


def _trace_seq(I, endP, dir_map):
    """Walk a crack skeleton sequentially, handling junctions via endpoint snapping.

    Processes endpoints one at a time.  At each step the pixel is marked as
    visited (zeroed) and the unique unvisited neighbour is followed.  When
    multiple unvisited neighbours are found (a junction), false-intersection
    pruning checks whether any of them still have onward connections; if only
    one remains the walk continues, otherwise the segment is terminated and the
    nearest remaining endpoint is snapped to as the segment's ``EndPoint``.
    Modifies ``I`` in place.

    Parameters
    ----------
    I : numpy.ndarray, 2-D, int (0/1)
        Skeleton image.  Modified in place as pixels are visited.
    endP : numpy.ndarray, shape (E, 2), int
        Endpoint coordinates ``(row, col)``, 0-based.
    dir_map : numpy.ndarray, shape (8, 2), int
        8-connectivity direction offsets.

    Returns
    -------
    list of numpy.ndarray, shape (M, 2), int
        Raw crack segments (before down-sampling).
    """
    crackRaw = []
    while endP.shape[0] > 0:
        start = endP[0].copy()
        row, col = int(start[0]), int(start[1])
        b = _nbr_vec(I, row, col)
        number = sum(b)
        if not (number >= 2):
            endP = endP[1:]
            I[row, col] = 0
        else:
            endP = np.roll(endP, -1, axis=0)

        tempX = []
        crack = []
        EndPoint = np.array([row, col], dtype=int)

        while True:
            tempX.append(row)
            b = _nbr_vec(I, row, col)
            number = sum(b)

            if number >= 2:
                nextall = [k for k in range(8) if b[k]]
                _touched = []
                for jdir in nextall:
                    rb = row + dir_map[jdir, 0]
                    cb = col + dir_map[jdir, 1]
                    _touched.append((rb, cb, I[rb, cb]))
                    I[rb, cb] = 0
                deleteInd = []
                for jj, jdir in enumerate(nextall):
                    rb = row + dir_map[jdir, 0]
                    cb = col + dir_map[jdir, 1]
                    bb = _nbr_vec(I, rb, cb)
                    if not any(bb):
                        deleteInd.append(jj)
                nextall = [d for k, d in enumerate(nextall) if k not in deleteInd]

                if len(nextall) == 1:
                    jdir = nextall[0]
                    row = row + dir_map[jdir, 0]
                    col = col + dir_map[jdir, 1]
                    crack.append([row, col])
                    continue
                for rb, cb, v in _touched:
                    I[rb, cb] = v
                if len(tempX) == 1:
                    break
                if endP.shape[0] > 0:
                    f, d = _dsearchn(endP, [row, col])
                    EndPoint = endP[f].copy() if d < 10 else np.array([row, col], dtype=int)
                else:
                    EndPoint = np.array([row, col], dtype=int)
                I[EndPoint[0], EndPoint[1]] = 1
                if (EndPoint[0] == start[0] and EndPoint[1] == start[1]) \
                        or spdist(EndPoint, [start])[0] < 2:
                    break
                b = _nbr_vec(I, row, col)
                number = sum(b)
                if not (number >= 2):
                    endP = endP[1:]
                break

            if number == 1:
                if endP.shape[0] > 0 and not (start[0] == row and start[1] == col):
                    f = (endP == [row, col]).all(axis=1)
                    if f.any():
                        endP = endP[~f]
                ind = b.index(1)
                row = row + dir_map[ind, 0]
                col = col + dir_map[ind, 1]
                crack.append([row, col])
                I[row, col] = 0

            if number == 0:
                if endP.shape[0] > 0:
                    f, d = _dsearchn(endP, [row, col])
                    EndPoint = endP[f].copy() if d < 10 else np.array([row, col], dtype=int)
                else:
                    EndPoint = np.array([row, col], dtype=int)
                if (EndPoint[0] == start[0] and EndPoint[1] == start[1]) \
                        or spdist(EndPoint, [start])[0] < 2:
                    break
                rb, cb = int(EndPoint[0]), int(EndPoint[1])
                b = _nbr_vec(I, rb, cb)
                ind = (endP == EndPoint).all(axis=1) if endP.shape[0] else np.array([], dtype=bool)
                if ind.any() and sum(b) == 0:
                    endP = endP[~ind]
                break

        if crack:
            seg = np.vstack(([start], np.asarray(crack, dtype=int), [EndPoint]))
            crackRaw.append(seg)
    return crackRaw


def _post(crackRaw):
    if not crackRaw:
        return [], np.empty((0, 4)), np.empty((0, 24)), np.empty((0, 24))
    line = np.zeros((len(crackRaw), 4), dtype=int)
    pointX = np.zeros((len(crackRaw), 24))
    pointY = np.zeros((len(crackRaw), 24))
    for o, cr in enumerate(crackRaw):
        line[o] = [cr[0, 0], cr[0, 1], cr[-1, 0], cr[-1, 1]]
        tempX = cr[:, 0]
        tempY = cr[:, 1]
        L = len(tempY)
        idx = mround(np.linspace(1, L, 24), 0).astype(int) - 1   # 1-based -> 0-based
        XI = tempX[idx].astype(float)
        YI = tempY[idx].astype(float)
        XI[0], YI[0] = tempX[0], tempY[0]
        XI[-1], YI[-1] = tempX[-1], tempY[-1]
        pointX[o] = XI
        pointY[o] = YI
    return crackRaw, line, pointX, pointY


def compCrack_branching(I, endP, dir_map, colors=None):
    """Trace a branchy crack skeleton into endpoint-to-branchpoint segments.

    Walks the skeleton starting from each endpoint.  When the walk reaches a
    genuine junction (a pixel with more than one onward neighbour after
    false-intersection pruning), the current segment is closed at that junction
    pixel and each unvisited outgoing branch is added to the work queue as a
    new segment start.  This decomposes the skeleton into a collection of
    simple segments whose endpoints are either true crack tips or branch
    points, which is the representation required by the offline crack-graph
    builder.

    The walk is performed on a zero-padded copy of the skeleton image so that
    border pixels always have well-defined (zero) out-of-bounds neighbours.

    Parameters
    ----------
    I : array-like, 2-D
        Binary crack skeleton; any non-zero value is foreground.
    endP : array-like, shape (E, 2), int
        Skeleton endpoint coordinates ``(row, col)``, 0-based.
    dir_map : array-like, shape (8, 2), int
        8-connectivity direction offsets ``(drow, dcol)``.
    colors : ignored
        Reserved parameter (unused); accepted for interface compatibility.

    Returns
    -------
    crackRaw : list of numpy.ndarray, shape (M, 2), int
        Ordered pixel coordinates for each segment (0-based).
    line : numpy.ndarray, shape (K, 4), int
        ``[start_row, start_col, end_row, end_col]`` for each segment.
    pointX : numpy.ndarray, shape (K, 24), float
        Each segment uniformly down-sampled to 24 row-coordinates.
    pointY : numpy.ndarray, shape (K, 24), float
        Corresponding 24 column-coordinates.
    """
    dir_map = np.asarray(dir_map, dtype=int)
    Ip = (np.pad(np.asarray(I) > 0, 1)).astype(int)         # zero border
    endP = np.atleast_2d(np.asarray(endP, dtype=int)).copy()
    endP = (endP + 1) if endP.size else np.empty((0, 2), dtype=int)   # padded coords

    def _b(r, c):
        return [int(Ip[r + dir_map[k, 0], c + dir_map[k, 1]]) for k in range(8)]

    crackRaw = []
    while endP.shape[0] > 0:
        start = endP[0].copy()
        row, col = int(start[0]), int(start[1])
        endP = endP[1:]
        Ip[row, col] = 0
        crack = []
        skp = True
        EndPoint = np.array([row, col], dtype=int)
        first = True
        while True:
            b = _b(row, col)
            number = sum(b)
            if number >= 2:                                  # intersection
                nextall = [k for k in range(8) if b[k]]
                for k in nextall:                            # zero immediate neighbours (v1 58-61)
                    Ip[row + dir_map[k, 0], col + dir_map[k, 1]] = 0
                delete = []                                  # false-intersection prune (64-78)
                for jj, k in enumerate(nextall):
                    rb, cb = row + dir_map[k, 0], col + dir_map[k, 1]
                    if not any(_b(rb, cb)):
                        delete.append(jj)
                nextall = [k for jj, k in enumerate(nextall) if jj not in delete]
                if len(nextall) == 1:                        # not a real intersection -> continue (79-87)
                    k = nextall[0]
                    row += dir_map[k, 0]; col += dir_map[k, 1]
                    crack.append([row, col])
                    continue
                EndPoint = np.array([row, col], dtype=int)   # real intersection (88-107)
                for k in nextall:                            # SPAWN each branch as a new start
                    nr, nc = row + dir_map[k, 0], col + dir_map[k, 1]
                    if endP.shape[0] and ((endP[:, 0] == nr) & (endP[:, 1] == nc)).any():
                        continue
                    endP = np.vstack([endP, [nr, nc]]) if endP.size else np.array([[nr, nc]])
                if (EndPoint == start).all() or abs(int((EndPoint - start).sum())) < 2:
                    skp = False
                break
            if number == 1:                                  # ongoing (159-169)
                k = b.index(1)
                row += dir_map[k, 0]; col += dir_map[k, 1]
                crack.append([row, col])
                Ip[row, col] = 0
            elif number == 0:                                # endpoint (170-214)
                EndPoint = np.array([row, col], dtype=int)
                if (EndPoint == start).all() or abs(int((EndPoint - start).sum())) < 2:
                    skp = False
                else:
                    if endP.shape[0]:
                        f = (endP[:, 0] == EndPoint[0]) & (endP[:, 1] == EndPoint[1])
                        if f.any():
                            endP = endP[~f]
                break
            first = False
        if crack and skp:                                    # record (217-224)
            seg = np.vstack(([start], np.asarray(crack, dtype=int), [EndPoint])) - 1
            crackRaw.append(seg)
    return _post(crackRaw)


def compCrack(I, endP, dir_map, colors=None, fg=None):
    """Trace a crack skeleton into ordered polylines (online-planner entry point).

    Dispatches to the sparse adjacency tracer when the skeleton consists of
    simple chains (no pixel with more than two 8-connected neighbours), and
    falls back to the sequential pixel-walk tracer otherwise.

    Parameters
    ----------
    I : array-like, 2-D
        Binary crack skeleton; any non-zero value is foreground.
    endP : array-like, shape (E, 2), int
        Skeleton endpoint coordinates ``(row, col)``, 0-based.
    dir_map : array-like, shape (8, 2), int
        8-connectivity direction offsets ``(drow, dcol)``.
    colors : ignored
        Reserved parameter (unused); accepted for interface compatibility.
    fg : numpy.ndarray, shape (N, 2), int, optional
        Precomputed foreground coordinates of ``I``.  When supplied, the
        function skips the full-image scan.  ``I`` must be 0/1 and ``fg``
        must list exactly its non-zero pixels.

    Returns
    -------
    crackRaw : list of numpy.ndarray, shape (M, 2), int
        Ordered pixel coordinates for each segment (0-based).
    line : numpy.ndarray, shape (K, 4), int
        ``[start_row, start_col, end_row, end_col]`` for each segment.
    pointX : numpy.ndarray, shape (K, 24), float
        Each segment uniformly down-sampled to 24 row-coordinates.
    pointY : numpy.ndarray, shape (K, 24), float
        Corresponding 24 column-coordinates.
    """
    Iarr = np.asarray(I)
    H, W = Iarr.shape
    endP = np.atleast_2d(np.asarray(endP, dtype=int)).copy()
    if endP.size == 0:
        endP = np.empty((0, 2), dtype=int)
    dir_map = np.asarray(dir_map, dtype=int)

    if fg is None:
        fg = argwhere2d(Iarr > 0)
    else:
        fg = np.atleast_2d(np.asarray(fg, dtype=int))
        if fg.size == 0:
            fg = np.empty((0, 2), dtype=int)

    crackRaw = _trace_fg(fg, endP, H, W, dir_map)
    if crackRaw is None:                 # a true junction (never in the OnlineSCC path)
        crackRaw = _trace_seq((Iarr > 0).astype(int), endP, dir_map)
    return _post(crackRaw)

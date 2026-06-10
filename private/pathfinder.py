"""Shortest obstacle-avoiding path planning for the crack-filling coverage planner.

Provides :func:`pathfinder`, which finds the shortest path between two points
inside a boundary polygon while staying within the polygon.  The path is
computed via a **visibility graph**: a weighted undirected graph whose nodes are
the start point, the boundary polygon vertices, and the end point, and whose
edges connect pairs of mutually visible nodes weighted by Euclidean distance.
The shortest path through this graph (Dijkstra via NetworkX) gives the
minimum-length obstacle-avoiding route.

This is used by the crack-planning module to route the robot from its current
position to each crack waypoint when a direct straight-line path would exit the
work area.
"""

from itertools import combinations

import numpy as np
import networkx as nx

from private.utils import spdist2
from .line_of_sight import line_of_sight


def pathfinder(start_point, end_point, external_boundaries):
    """Find the shortest obstacle-avoiding path between two points inside a boundary polygon.

    Builds a visibility graph over the start point, all boundary polygon
    vertices, and the end point.  An edge is added between each pair of nodes
    that have an unobstructed line of sight (tested via
    :func:`~private.line_of_sight.line_of_sight`), weighted by Euclidean
    distance.  The shortest path from start to end through this graph is
    returned.

    Parameters
    ----------
    start_point : array-like, shape (2,) or (3,)
        Starting position (x, y[, unused]).
    end_point : array-like, shape (2,) or (3,)
        Target position (x, y[, unused]).
    external_boundaries : array-like, shape (M, 2)
        Vertices of the boundary polygon (closed ring).

    Returns
    -------
    waypoint_coordinates : numpy.ndarray, shape (K, 2)
        Sequence of (x, y) waypoints from start to end, including both
        endpoints.
    weight : float
        Total Euclidean path length.
    """
    start_point = np.asarray(start_point, dtype=float).ravel()
    end_point = np.asarray(end_point, dtype=float).ravel()
    B = np.atleast_2d(np.asarray(external_boundaries, dtype=float))

    nodes = np.vstack([start_point[:2], B, end_point[:2]])
    N = nodes.shape[0]

    ii = np.array(list(combinations(range(N), 2)))           # 0-based node pairs
    w = spdist2(nodes[ii[:, 0], :2], nodes[ii[:, 1], :2])
    vis = line_of_sight(nodes[ii[:, 0], :2], nodes[ii[:, 1], :2], B)

    edges = ii[vis > 0]
    ew = w[vis > 0]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for (a, b), wt in zip(edges, ew):
        G.add_edge(int(a), int(b), weight=float(wt))

    path = nx.shortest_path(G, 0, N - 1, weight="weight")
    weight = nx.shortest_path_length(G, 0, N - 1, weight="weight")
    waypoint_coordinates = nodes[path][:, [0, 1]]
    return waypoint_coordinates, float(weight)

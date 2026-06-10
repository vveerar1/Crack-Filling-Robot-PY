"""Supporting routines for the crack-filling planners.

Helpers used by SCC and OnlineSCC: crack skeleton tracing (``compCrack``),
endpoint/branch-point detection (``endP_ident``, ``bwmorph``), visibility graph
(``line_of_sight``, ``pathfinder``), polygon utilities (``refinePoly``,
``DecimatePoly``, the ``Polygons_intersection_*`` suite), and plotting helpers
(``drawArrowHead``).
"""

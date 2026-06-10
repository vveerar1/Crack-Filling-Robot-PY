"""Polygon pairwise intersection for the crack-filling coverage planner.

Provides :func:`Polygons_Intersection`, which computes the membership arrangement
of a collection of polygons.  Given N input polygons, the function partitions
their union into disjoint atomic regions and labels each region by the subset of
input polygons that cover it.

This is used by the crack-planning module to identify sensor-footprint overlap
regions (those covered by two or more sensor disks), whose centroids serve as
nodes in the crack visibility graph.

The computation uses a Shapely planar overlay: all polygon boundaries are nodded,
polygonized into atomic faces, each face is labelled by the subset of input
polygons whose interior contains the face's representative point, and faces with
equal membership are merged.
"""

import numpy as np
from shapely.ops import unary_union, polygonize

from private import poly_utils as pu


def Polygons_Intersection(polys, display=0, accuracy=1e-3):
    """Compute the pairwise intersection arrangement of a list of polygons.

    Partitions the union of all input polygons into disjoint atomic regions and
    returns each region annotated with the set of input polygons that cover it.

    Parameters
    ----------
    polys : list of PolyShape or Shapely Polygon
        Input polygons (e.g. sensor-footprint disks).
    display : int, optional
        Unused; kept for API compatibility.
    accuracy : float, optional
        Unused; kept for API compatibility.

    Returns
    -------
    list of dict
        One entry per distinct overlap region, ordered by membership-set size
        then by membership indices.  Each dict has:

        ``'index'`` : list of int
            0-based indices into ``polys`` of the polygons that cover this region.
        ``'P'`` : Shapely geometry
            The merged region geometry.
        ``'area'`` : float
            Area of the region.
    """
    geoms = [p.geom if isinstance(p, pu.PolyShape) else p for p in polys]
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return []

    boundaries = unary_union([g.boundary for g in geoms])
    faces = list(polygonize(boundaries))

    groups = {}
    for f in faces:
        rp = f.representative_point()
        idx = tuple(i for i, g in enumerate(geoms) if g.contains(rp))
        if not idx:
            continue
        groups.setdefault(idx, []).append(f)

    Geo = []
    for idx in sorted(groups, key=lambda k: (len(k), k)):
        region = unary_union(groups[idx])
        Geo.append({"index": list(idx), "P": region, "area": region.area})
    return Geo

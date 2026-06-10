"""End-to-end smoke tests for the two planners on a standard crack map.

These run the planners exactly as a user would and assert that each produces a
sane coverage + crack-filling path. They are self-contained -- they need only the
bundled crack map under ``CrackMaps/`` (no external reference data).

- ``test_scc_offline`` is quick (~30 s).
- ``test_onlinescc_online`` runs the full online loop (a few minutes); it is
  marked ``slow`` so ``pytest -m "not slow"`` skips it.

Bounds are loose enough to absorb the small, platform-dependent variation in
the skeletonization step (which shifts crack-detection timing by a pose or two)
while still catching any real regression.
"""
import numpy as np
import pytest

_MAP = "myCrack8_100_1"


def test_scc_offline():
    """Offline SCC: plan a coverage + filling route over the fully-known map."""
    from SCC import run_scc

    PathEdge, res = run_scc(_MAP, dd=8)
    coverage_pct, _density, _t, path_len_ft, area_ft2 = res

    assert PathEdge.ndim == 2 and PathEdge.shape[1] == 2
    assert PathEdge.shape[0] > 100                 # a non-trivial route
    assert coverage_pct > 90.0                     # near-complete sensor coverage
    assert 150.0 < path_len_ft < 162.0             # route length (ft), default rpp route
    assert 488.0 < area_ft2 < 502.0                # area covered (sq ft)
    assert np.isfinite(PathEdge).all()


@pytest.mark.slow
def test_onlinescc_online():
    """Online OnlineSCC: scan-and-fill the map discovered while moving."""
    from OnlineSCC import run_online_scc

    PathEdge, res = run_online_scc(_MAP, dd=8)
    num_iter, _density, _t, path_len_ft, area_ft2 = res

    assert PathEdge.ndim == 2 and PathEdge.shape[1] == 2
    assert PathEdge.shape[0] > 300                 # full sweep + fills
    assert 360 < num_iter < 410                    # scan/fill iterations
    assert 155.0 < path_len_ft < 164.0             # route length (ft)
    assert 498.0 < area_ft2 < 511.0                # area covered (sq ft)
    assert np.isfinite(PathEdge).all()

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-11

First public release — the full Python implementation of both planners and the shared coverage
pipeline.

### Added
- **OnlineSCC** — online planner for *unknown* crack locations: boustrophedon (zig-zag) scanning
  with on-the-fly crack detection (skeleton endpoints / branch points), visibility-graph +
  Chinese Postman crack-fill planning, and iterative re-decomposition of the remaining free space
  until coverage is complete.
- **SCC** — offline planner for *known* crack maps: up-front crack-graph extraction combined with
  the coverage graph and routed with a modified Chinese Postman (Eulerian routing, with
  integer-programming matching of odd-degree nodes when needed).
- Shared coverage pipeline: Morse cell decomposition (MCD), Reeb graph, Reeb-path cell ordering,
  and boustrophedon path generation with cell-connection for disjoint regions.
- Geometry and imaging support: Shapely-backed polygon utilities, polygon decimation, skeleton
  morphology, visibility / line-of-sight tests, the crack-skeleton tracer, and numerical helpers.
- Visualization: per-step figures and a standalone animated-GIF renderer (`python -m private.make_gif`)
  of the robot scanning and filling.
- Configurable robot dimensions via `robot_config.json`, CLI map selection (`--den` / `--map-num` /
  `--sig`), and SCC route selection (`--route`, default the geometry-routed `rpp`).
- Crack-map dataset under `CrackMaps/` (uniform and Gaussian-distributed maps across densities).
- End-to-end smoke tests for both planners (`tests/`).

[1.0.0]: https://github.com/vveerar1/Crack-Filling-Robot-PY/releases/tag/v1.0.0
